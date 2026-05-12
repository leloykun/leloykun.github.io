"""Generated generic TileLang source for a materialized OnlinePlan summary."""
import argparse
import tilelang
import tilelang.language as T

def get_configs():
    import itertools
    max_shared_mem = 163840  # A100 SXM4 dynamic shared-memory limit per CTA.
    warp_alignment = 4
    block_options = {
        'block_m': [16, 32, 64, 128],
        'block_d': [16, 32, 64, 128],
        'block_n': [128, 256, 512, 1024, 2048, 4096],
    }
    base = [
        {"threads": threads, "num_stages": num_stages, "enable_swizzle": enable_swizzle}
        for threads, num_stages, enable_swizzle in itertools.product(
            [128, 256],
            [1, 2, 3],
            [True, False],
        )
    ]
    configs = []
    keys = list(block_options)
    for values in itertools.product(*(block_options[key] for key in keys)):
        for config in base:
            item = dict(config)
            item.update(dict(zip(keys, values)))
            threads = int(item["threads"])
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
            block_row = int(item['block_m'])
            block_col = int(item['block_n'])
            block_stream = int(item['block_d'])
            scale_shared = 2 * block_row * block_stream + 4 * block_row
            compute_shared = 2 * block_row * block_stream + 4 * block_stream * block_col
            shared_mem = int(item["num_stages"]) * max(scale_shared, compute_shared)
            if shared_mem > max_shared_mem:
                continue
            configs.append(item)
    return configs

def build_kernel(
    d: int = 1024,
    m: int = 1024,
    n: int = 1024,
    block_d: int = 64,
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
    scale_jit_decorator = tilelang.jit(out_idx=[2], pass_configs=fast_math_pass_configs)
    compute_jit_decorator = tilelang.jit(out_idx=[4], pass_configs=fast_math_pass_configs)
    if enable_autotune:
        def decorate_scale(fn):
            return tilelang.autotune(configs=get_configs(), warmup=autotune_warmup, rep=autotune_rep, timeout=autotune_timeout, skip_check=True)(scale_jit_decorator(fn))
        def decorate_compute(fn):
            return tilelang.autotune(configs=get_configs(), warmup=autotune_warmup, rep=autotune_rep, timeout=autotune_timeout, skip_check=True)(compute_jit_decorator(fn))
    else:
        def decorate_scale(fn):
            return scale_jit_decorator(fn)
        def decorate_compute(fn):
            return compute_jit_decorator(fn)

    @decorate_scale
    def rmsnorm_mlp_c2177_scale_jit(d: int = d, m: int = m, n: int = n, block_d: int = block_d, block_m: int = block_m, block_n: int = block_n, threads: int = threads, num_stages: int = num_stages, enable_swizzle: bool = enable_swizzle):
        @T.prim_func
        def scale_main(
            X: T.Tensor((m, d), dtype),
            eps: T.float32,
            S: T.Tensor((m,), accum_dtype),
        ):
            with T.Kernel(T.ceildiv(m, block_m), threads=threads) as gy:
                X_tile = T.alloc_shared((block_m, block_d), dtype)
                X_pow = T.alloc_fragment((block_m, block_d), accum_dtype)
                row_state = T.alloc_fragment((block_m,), accum_dtype)
                T.clear(X_pow)
                for k_stream in T.Pipelined(T.ceildiv(d, block_d), num_stages=num_stages):
                    T.copy(X[gy * block_m, k_stream * block_d], X_tile)
                    for i0, i1 in T.Parallel(block_m, block_d):
                        X_pow[i0, i1] = X_pow[i0, i1] + X_tile[i0, i1] * X_tile[i0, i1]
                T.reduce_sum(X_pow, row_state, dim=1)
                for i0 in T.Parallel(block_m):
                    S[gy * block_m + i0] = 1.0 / T.sqrt(row_state[i0] / d + eps)
        return scale_main

    @decorate_compute
    def rmsnorm_mlp_c2177_compute_jit(d: int = d, m: int = m, n: int = n, block_d: int = block_d, block_m: int = block_m, block_n: int = block_n, threads: int = threads, num_stages: int = num_stages, enable_swizzle: bool = enable_swizzle):
        @T.prim_func
        def compute_main(
            X: T.Tensor((m, d), dtype),
            W_gate: T.Tensor((d, n), dtype),
            W_up: T.Tensor((d, n), dtype),
            S: T.Tensor((m,), accum_dtype),
            O: T.Tensor((m, n), dtype),
        ):
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=threads) as (gx, gy):
                X_tile = T.alloc_shared((block_m, block_d), dtype)
                W_gate_tile = T.alloc_shared((block_d, block_n), dtype)
                acc_0 = T.alloc_fragment((block_m, block_n), accum_dtype)
                W_up_tile = T.alloc_shared((block_d, block_n), dtype)
                acc_1 = T.alloc_fragment((block_m, block_n), accum_dtype)
                T.use_swizzle(panel_size=10, enable=enable_swizzle)
                T.clear(acc_0)
                T.clear(acc_1)
                for k_stream in T.Pipelined(T.ceildiv(d, block_d), num_stages=num_stages):
                    T.copy(X[gy * block_m, k_stream * block_d], X_tile)
                    T.copy(W_gate[k_stream * block_d, gx * block_n], W_gate_tile)
                    T.gemm(X_tile, W_gate_tile, acc_0, clear_accum=False)
                    T.copy(W_up[k_stream * block_d, gx * block_n], W_up_tile)
                    T.gemm(X_tile, W_up_tile, acc_1, clear_accum=False)
                for i0, i1 in T.Parallel(block_m, block_n):
                    acc_0[i0, i1] = acc_0[i0, i1] * S[gy * block_m + i0]
                    acc_1[i0, i1] = acc_1[i0, i1] * S[gy * block_m + i0]
                    acc_0[i0, i1] = acc_0[i0, i1] * acc_1[i0, i1]
                T.copy(acc_0, O[gy * block_m, gx * block_n])
        return compute_main

    scale_kernel = rmsnorm_mlp_c2177_scale_jit()
    compute_kernel = rmsnorm_mlp_c2177_compute_jit()
    input_names = ['X', 'W_gate', 'W_up']
    def materialized_kernel(*args):
        input_count = len(input_names)
        values = dict(zip(input_names, args[:input_count]))
        eps_value = args[input_count] if len(args) > input_count else 1.0e-5
        S = scale_kernel(values['X'], eps_value)
        return compute_kernel(values['X'], values['W_gate'], values['W_up'], S)
    def get_kernel_source():
        parts = []
        for kernel in (scale_kernel, compute_kernel):
            if hasattr(kernel, 'get_kernel_source'):
                parts.append(str(kernel.get_kernel_source()))
        return '\n\n'.join(parts)
    materialized_kernel.get_kernel_source = get_kernel_source
    return materialized_kernel

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
