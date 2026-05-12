"""Generated generic TileLang source for a ShardedDAG candidate."""
import argparse
import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout as make_swizzle_layout

def get_configs():
    import itertools
    max_shared_mem = 163840  # A100 SXM4 dynamic shared-memory limit per CTA.
    warp_alignment = 8
    base = [
        {"threads": threads, "num_stages": num_stages, "enable_swizzle": enable_swizzle}
        for threads, num_stages, enable_swizzle in itertools.product(
            [128, 256],
            [1, 2, 3],
            [True, False],
        )
    ]
    block_options = {'block_d': [32, 64, 128, 256, 512], 'block_m': [32, 64, 128, 256, 512], 'block_n': [32, 64, 128, 256, 512]}
    configs = []
    keys = list(block_options)
    for values in itertools.product(*(block_options[key] for key in keys)):
        for config in base:
            item = dict(config)
            item.update(dict(zip(keys, values)))
            threads = int(item['threads'])
            if threads % 32 != 0:
                continue
            warp_count = threads // 32
            warp_ok = True
            for key in keys:
                block = int(item[key])
                if (block // warp_count) % warp_alignment != 0:
                    warp_ok = False
                    break
            if not warp_ok:
                continue
            shared_mem = int(item['num_stages']) * ((2 * (int(item['block_m']) * int(item['block_d']))) + (2 * (int(item['block_d']) * int(item['block_n']))) + (2 * (int(item['block_d']) * int(item['block_n']))))
            if shared_mem > max_shared_mem:
                continue
            configs.append(item)
    return configs

def build_kernel(
    d: int = 2048,
    m: int = 1024,
    n: int = 2048,
    block_d: int = 128,
    block_m: int = 128,
    block_n: int = 128,
    threads: int = 256,
    num_stages: int = 2,
    enable_swizzle: bool = True,
    enable_autotune: bool = False,
    autotune_warmup: int = 10,
    autotune_rep: int = 10,
    autotune_timeout: int = 100,
):
    dtype = T.float16
    accum_dtype = T.float32
    scale = 1.44269504  # log2(e)
    fast_math_pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    jit_decorator = tilelang.jit(out_idx=[-1], pass_configs=fast_math_pass_configs)
    if enable_autotune:
        def decorate(fn):
            return tilelang.autotune(configs=get_configs(), warmup=autotune_warmup, rep=autotune_rep, timeout=autotune_timeout, skip_check=True)(jit_decorator(fn))
    else:
        def decorate(fn):
            return jit_decorator(fn)

    @decorate
    def swiglu_c517_jit(d: int = d, m: int = m, n: int = n, block_d: int = block_d, block_m: int = block_m, block_n: int = block_n, threads: int = threads, num_stages: int = num_stages, enable_swizzle: bool = enable_swizzle):
        @T.prim_func
        def main(
            X: T.Tensor((m, d), dtype),
            W_up: T.Tensor((d, n), dtype),
            W_gate: T.Tensor((d, n), dtype),
            O: T.Tensor((m, n), dtype),
        ):
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (gx, gy):
                input_0 = T.alloc_shared((block_m, block_d), dtype)
                input_4 = T.alloc_shared((block_d, block_n), dtype)
                matmul_8 = T.alloc_fragment((block_m, block_n), accum_dtype)
                input_10 = T.alloc_shared((block_d, block_n), dtype)
                matmul_14 = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.annotate_layout({
                    input_0: make_swizzle_layout(input_0),
                    input_4: make_swizzle_layout(input_4),
                    input_10: make_swizzle_layout(input_10),
                })
                T.use_swizzle(panel_size=10, enable=enable_swizzle)
                T.clear(matmul_8)
                T.clear(matmul_14)
                for k_pass0 in T.Pipelined(T.ceildiv(d, block_d), num_stages=num_stages):
                    T.copy(X[gy * block_m, k_pass0 * block_d], input_0)
                    T.copy(W_up[k_pass0 * block_d, gx * block_n], input_4)
                    T.gemm(input_0, input_4, matmul_8, clear_accum=False)
                    T.copy(W_gate[k_pass0 * block_d, gx * block_n], input_10)
                    T.gemm(input_0, input_10, matmul_14, clear_accum=False)
                for i0, i1 in T.Parallel(block_m, block_n):
                    matmul_14[i0, i1] = matmul_14[i0, i1] / (1.0 + T.exp2((-(matmul_14[i0, i1])) * scale))
                    matmul_14[i0, i1] = matmul_8[i0, i1] * matmul_14[i0, i1]
                T.copy(matmul_14, O[gy * block_m, gx * block_n])
        return main

    return swiglu_c517_jit()

def build_autotuned_kernel(warmup: int = 10, rep: int = 10, timeout: int = 100):
    return build_kernel(enable_autotune=True, autotune_warmup=warmup, autotune_rep=rep, autotune_timeout=timeout)


def launch(*args):
    kernel = build_kernel()
    return kernel(*args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-tilelang", action="store_true")
    args = parser.parse_args()
    if args.check_tilelang:
        print("tilelang-ok")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
