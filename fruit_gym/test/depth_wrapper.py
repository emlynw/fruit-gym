# --- keep your existing imports ---
import os, sys, importlib, types, traceback
import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F
from collections.abc import Iterable
from typing import Optional, Dict, Any
import time
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------
# Import VDA as a package (no install needed) from a directory
# ---------------------------------------------------------------------
def _import_vda_stream_from_pkg_dir(pkg_dir: str):
    """
    Make 'video_depth_anything' importable from `pkg_dir` and return
    video_depth_anything.video_depth_stream.VideoDepthAnything
    """
    if not pkg_dir or not os.path.isdir(pkg_dir):
        raise FileNotFoundError(f"VDA package directory not found: {pkg_dir}")

    pkg_name = "video_depth_anything"
    parent = os.path.dirname(pkg_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [pkg_dir]
        sys.modules[pkg_name] = pkg

    mod = importlib.import_module(f"{pkg_name}.video_depth_stream")
    if not hasattr(mod, "VideoDepthAnything"):
        raise ImportError("VideoDepthAnything not found in video_depth_stream")
    return mod.VideoDepthAnything


class VideoDepthEstimator:
    def __init__(self, device: str = "cuda", input_size: int = 384):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = int(input_size)
        self._state = {}

    def reset_stream(self, key: str) -> None:
        self._state.pop(key, None)

    def __call__(self, frame_bgr_or_rgb: np.ndarray, key: str) -> np.ndarray:
        raise NotImplementedError


class VideoDepthAnythingAdapter(VideoDepthEstimator):
    def __init__(
        self,
        device: str = "cuda",
        input_size: int = 384,
        model_name: str = "vits",           # vits | vitb | vitl
        weights_path: str | None = None,    # <-- allow None = auto-pick
        weights_dir: str = "/home/emlyn/video_depth_anything_models",
        use_metric: bool = False,
        vda_pkg_dir: str = "/home/emlyn/Video-Depth-Anything/video_depth_anything",
        verbose_import_errors: bool = True,
        fp32: bool = False,
    ):
        super().__init__(device=device, input_size=input_size)
        self.model_name = model_name
        self.use_metric = use_metric
        self.return_unit_range = not use_metric
        self.vda_pkg_dir = vda_pkg_dir
        self.verbose_import_errors = verbose_import_errors
        self.fp32 = bool(fp32)

        # decide weights
        if weights_path is None:
            prefix = "metric_" if self.use_metric else ""
            fname = f"{prefix}video_depth_anything_{self.model_name}.pth"
            self.weights_path = os.path.join(weights_dir, fname)
        else:
            self.weights_path = weights_path

        self._model = None
        self._backend = None  # "vda-stream"

        # per-repo config for streaming class (matches your vits checkpoint)
        self._model_configs = {
            "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        # Cudagraph support
        self.enable_cudagraph = (torch.cuda.is_available() and not self.fp32)
        self._cg = None
        self._cg_shape = None               # (B,T,C,H,W)
        self._static_in = None              # tensor captured by graph
        self._static_feat = None            # output placeholder from graph

    # ---- build the streaming model with weights that MATCH the head sizes ----
    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            VideoDepthAnything = _import_vda_stream_from_pkg_dir(self.vda_pkg_dir)
            self._build_vda_stream(VideoDepthAnything)
            print(f"[DepthAdapter] Backend: vda-stream (pkg_dir={self.vda_pkg_dir})")
        except Exception:
            if self.verbose_import_errors:
                print("[DepthAdapter] VDA-stream import/build failed:")
                traceback.print_exc()
            raise

    def _build_vda_stream(self, VideoDepthAnything):
        if self.model_name not in self._model_configs:
            raise ValueError(f"Unknown encoder '{self.model_name}'")
        cfg = self._model_configs[self.model_name]

        # Instantiate streaming model (expects small head for vits)
        self._model = VideoDepthAnything(**cfg)

        # Pick the right small checkpoint (metric or relative)
        if not os.path.isfile(self.weights_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.weights_path}")

        state = torch.load(self.weights_path, map_location="cpu")
        # Streaming example uses strict=True (weights must match head sizes)
        self._model.load_state_dict(state, strict=True)

        try:
            self._model.to(self.device)
        except Exception:
            pass
        self._model.eval()
        self._backend = "vda-stream"

    @torch.no_grad()
    def _capture_cudagraph(self, cur_input: torch.Tensor):
        if not self.enable_cudagraph or self.device.type != "cuda":
            return

        shape = tuple(cur_input.shape)
        if self._cg is not None and self._cg_shape == shape:
            return

        # static input buffer (keep addresses stable)
        self._static_in = torch.empty_like(cur_input, device=self.device).contiguous()
        torch.cuda.synchronize()

        # warmup under autocast (initializes kernels/algos in the same dtype as capture)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(not self.fp32)):
            _ = self._model.forward_features(self._static_in)

        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        self._static_feat = None

        with torch.cuda.graph(g):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(not self.fp32)):
                out = self._model.forward_features(self._static_in)
            self._static_feat = out

        self._cg = g
        self._cg_shape = shape

    @torch.no_grad()
    def _infer_once_cg(self, frame_rgb_uint8: np.ndarray) -> np.ndarray:
        """
        Use the VDA streaming transform/caches, but run forward_features under a CUDA graph.
        Assumes the first call went through _infer_once(), which initializes m.transform & caches.
        """
        m = self._model

        # If transform/caches aren't initialized yet, fall back once to the repo path.
        if m.transform is None:
            return self._infer_once(frame_rgb_uint8)

        # ---- Preprocess (same as repo) ----
        frame = frame_rgb_uint8
        frame_h, frame_w = frame.shape[:2]
        assert frame_h == m.frame_height and frame_w == m.frame_width

        img_f32 = frame.astype(np.float32) / 255.0
        tdict = m.transform({'image': img_f32})
        cur_input = torch.from_numpy(tdict['image']).unsqueeze(0).unsqueeze(0)  # [1,1,3,H',W']
        cur_input = cur_input.to(self.device, non_blocking=True).contiguous()

        # Feed FP32 into the capture to avoid half/float bias mismatch in Conv2d
        # (If you want FP16 capture, wrap forward_features in autocast inside _capture_cudagraph)
        if cur_input.dtype != torch.float32:
            cur_input = cur_input.float()

        # ---- Encoder with CUDA Graph (captured on first size) ----
        self._capture_cudagraph(cur_input)
        self._static_in.copy_(cur_input)
        self._cg.replay()
        cur_feature = self._static_feat
        x_shape = cur_input.shape

        # ---- Temporal cache (no hidden attrs; infer window length from existing cache) ----
        infer_len = len(m.frame_cache_list)  # set by the first eager call
        if infer_len >= 4:
            # mimic repo pattern: first 2 + last (infer_len-3)
            cur_list = m.frame_cache_list[0:2] + m.frame_cache_list[-infer_len + 3:]
        else:
            # very early frames: use whatever is available (or None)
            cur_list = m.frame_cache_list

        cur_cache = None
        if cur_list:
            # cur_list is a list of lists of tensors; stitch per index
            cur_cache = [torch.cat([h[i] for h in cur_list], dim=1) for i in range(len(cur_list[0]))]

        # ---- Depth head (keep autocast for speed if fp16 allowed) ----
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(not self.fp32)):
            if cur_cache is None:
                depth, new_cache = m.forward_depth(cur_feature, x_shape)
            else:
                depth, new_cache = m.forward_depth(cur_feature, x_shape, cached_hidden_state_list=cur_cache)

        # ---- Resize back to original frame size ----
        depth = depth.to(cur_input.dtype)
        depth = F.interpolate(
            depth.flatten(0, 1).unsqueeze(1),
            size=(frame_h, frame_w),
            mode='bilinear',
            align_corners=True
        )
        new_depth = depth[-1, 0].float().cpu().numpy()

        # ---- Slide window: keep cache length constant (equal to initial infer_len) ----
        m.frame_cache_list.append(new_cache)
        if infer_len > 0 and len(m.frame_cache_list) > infer_len:
            # drop the oldest AFTER the first element (matching repo’s behavior)
            del m.frame_cache_list[1]

        d = new_depth.astype(np.float32)
        if self.return_unit_range:
            dmin, dmax = float(d.min()), float(d.max())
            d = (d - dmin) / (dmax - dmin + 1e-8) if dmax > dmin else np.zeros_like(d, dtype=np.float32)
        return d

    @torch.no_grad()
    def _infer_once(self, rgb_uint8_hwc: np.ndarray) -> np.ndarray:
        if rgb_uint8_hwc.dtype != np.uint8 or not rgb_uint8_hwc.flags['C_CONTIGUOUS']:
            rgb_uint8_hwc = np.ascontiguousarray(np.clip(rgb_uint8_hwc, 0, 255).astype(np.uint8))

        out = self._model.infer_video_depth_one(
            rgb_uint8_hwc,
            input_size=int(self.input_size),   # let their cached transform do the resize
            device=("cuda" if self.device.type == "cuda" else "cpu"),
            fp32=self.fp32,
        )
        d = out if isinstance(out, np.ndarray) else np.array(out)
        if d.ndim == 3 and d.shape[2] == 3 and d.dtype == np.uint8:
            print("Warning: depth output looks like a 3-channel uint8 image; converting to grayscale float")
            d = (0.2989*d[...,0] + 0.5870*d[...,1] + 0.1140*d[...,2]).astype(np.float32)/255.0
        else:
            d = d.astype(np.float32)
            if d.max() > 1.5 and d.max() <= 255.0:
                d = d/255.0
        return np.clip(d, 0.0, 1.0)

    def __call__(self, frame_bgr_or_rgb: np.ndarray, key: str) -> np.ndarray:
        self._lazy_init()
        img = frame_bgr_or_rgb
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image for key '{key}', got {img.shape}.")

        if self.enable_cudagraph and self.device.type == "cuda":
            return self._infer_once_cg(img)
        else:
            return self._infer_once(img)

    def reset_stream(self, key: str) -> None:
        # streaming model keeps its own temporal state; if it exposes a reset,
        # call it here. Otherwise, do nothing.
        if hasattr(self._model, "reset_stream"):
            try:
                self._model.reset_stream()
            except Exception:
                pass


# ---------------- Gym wrapper ----------------
class VideoDepthObsWrapper(gym.ObservationWrapper):
    """
    Adds depth maps for specified RGB obs keys as obs[f"{key}_depth"] (float32 HxW).
    """
    def __init__(
        self,
        env: gym.Env,
        depth_estimator: VideoDepthEstimator,
        rgb_keys: Iterable[str],
        normalize_depth: bool = True,
    ):
        super().__init__(env)
        self.depth_estimator = depth_estimator
        self.rgb_keys = tuple(rgb_keys)
        self.normalize_depth = normalize_depth

        assert isinstance(env.observation_space, gym.spaces.Dict), \
            "VideoDepthObsWrapper expects a Dict observation space."

        new_spaces = dict(env.observation_space.spaces)
        for key in self.rgb_keys:
            if key not in new_spaces:
                raise KeyError(f"Key '{key}' not found in env.observation_space.")
            space = new_spaces[key]
            if not (isinstance(space, gym.spaces.Box) and space.shape and len(space.shape) == 3 and space.shape[2] == 3):
                raise TypeError(f"Observation '{key}' must be HxWx3 image Box.")
            H, W, _ = space.shape
            if self.normalize_depth:
                depth_space = gym.spaces.Box(low=0.0, high=1.0, shape=(H, W), dtype=np.float32)
            else:
                depth_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(H, W), dtype=np.float32)
            new_spaces[f"{key}_depth"] = depth_space

        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        obs = dict(observation)
        depth_start_time = time.time()
        for key in self.rgb_keys:
            rgb = obs[key]
            if rgb.dtype != np.uint8:
                if rgb.dtype in (np.float16, np.float32, np.float64):
                    scale = 255.0 if rgb.max() <= 1.0 else 1.0
                    rgb_uint8 = np.clip(rgb * scale, 0, 255).astype(np.uint8)
                else:
                    raise TypeError(f"Obs '{key}' must be uint8 or float; got {rgb.dtype}")
            else:
                rgb_uint8 = rgb
            if not rgb_uint8.flags['C_CONTIGUOUS'] or any(s < 0 for s in rgb_uint8.strides):
                rgb_uint8 = np.ascontiguousarray(rgb_uint8)

            depth = self.depth_estimator(rgb_uint8, key=key).astype(np.float32)
            if self.normalize_depth:
                dmin, dmax = float(depth.min()), float(depth.max())
                depth = (depth - dmin) / (dmax - dmin + 1e-8) if dmax > dmin else np.zeros_like(depth, dtype=np.float32)
            obs[f"{key}_depth"] = depth
        print(f"[DepthWrapper] Depth inference time: {time.time() - depth_start_time:.3f}s")
        return obs

    def reset(self, **kwargs):
        for key in self.rgb_keys:
            self.depth_estimator.reset_stream(key)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info
