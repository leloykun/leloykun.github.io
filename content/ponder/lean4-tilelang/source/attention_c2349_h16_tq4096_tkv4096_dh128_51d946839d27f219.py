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
    block_options = {'block_t_kv': [32, 64, 128, 256, 512], 'block_t_q': [32, 64, 128, 256, 512]}
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
            shared_mem = int(item['num_stages']) * ((2 * (int(item['block_t_q']) * 128)) + (2 * (int(item['block_t_kv']) * 128)) + (2 * (int(item['block_t_kv']) * 128)))
            if shared_mem > max_shared_mem:
                continue
            configs.append(item)
    return configs

def build_kernel(
    d_h: int = 128,
    h: int = 16,
    t_kv: int = 4096,
    t_q: int = 4096,
    block_t_kv: int = 128,
    block_t_q: int = 128,
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
    def attention_c2349_jit(d_h: int = d_h, h: int = h, t_kv: int = t_kv, t_q: int = t_q, block_t_kv: int = block_t_kv, block_t_q: int = block_t_q, threads: int = threads, num_stages: int = num_stages, enable_swizzle: bool = enable_swizzle):
        @T.prim_func
        def main(
            Q: T.Tensor((h, t_q, d_h), dtype),
            K: T.Tensor((h, t_kv, d_h), dtype),
            V: T.Tensor((h, t_kv, d_h), dtype),
            O: T.Tensor((h, t_q, d_h), dtype),
        ):
            with T.Kernel(T.ceildiv(t_q, block_t_q), h, threads=threads) as (gx, gy):
                input_0 = T.alloc_shared((block_t_q, d_h), dtype)
                input_4 = T.alloc_shared((block_t_kv, d_h), dtype)
                matmul_8 = T.alloc_fragment((block_t_q, block_t_kv), accum_dtype)
                red_max_9 = T.alloc_fragment((block_t_q,), accum_dtype)
                input_15 = T.alloc_shared((block_t_kv, d_h), dtype)
                matmul_19 = T.alloc_fragment((block_t_q, d_h), accum_dtype)
                state_pass0_o = T.alloc_fragment((block_t_q, d_h), accum_dtype)
                red_sum_22 = T.alloc_fragment((block_t_q,), accum_dtype)
                state_pass0_l = T.alloc_fragment((block_t_q,), accum_dtype)
                state_pass0_m = T.alloc_fragment((block_t_q,), accum_dtype)
                scale_old_state_pass0_m = T.alloc_fragment((block_t_q,), accum_dtype)
                scale_tile_state_pass0_m = T.alloc_fragment((block_t_q,), accum_dtype)
                cast_lhs_19 = T.alloc_fragment((block_t_q, block_t_kv), dtype)
                T.annotate_layout({
                    input_0: make_swizzle_layout(input_0),
                    input_4: make_swizzle_layout(input_4),
                    input_15: make_swizzle_layout(input_15),
                })
                T.use_swizzle(panel_size=10, enable=enable_swizzle)
                T.fill(state_pass0_m, -T.infinity(accum_dtype))
                T.clear(state_pass0_l)
                T.clear(state_pass0_o)
                T.copy(Q[gy, gx * block_t_q, 0], input_0)
                for k_pass0 in T.Pipelined(T.ceildiv(t_kv, block_t_kv), num_stages=num_stages):
                    T.copy(K[gy, k_pass0 * block_t_kv, 0], input_4)
                    T.gemm(input_0, input_4, matmul_8, clear_accum=True, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.reduce_max(matmul_8, red_max_9, dim=1, clear=True)
                    for i0, i1 in T.Parallel(block_t_q, block_t_kv):
                        matmul_8[i0, i1] = matmul_8[i0, i1] - red_max_9[i0]
                        matmul_8[i0, i1] = T.exp2((matmul_8[i0, i1]) * scale)
                    T.reduce_sum(matmul_8, red_sum_22, dim=1, clear=True)
                    T.copy(V[gy, k_pass0 * block_t_kv, 0], input_15)
                    T.copy(matmul_8, cast_lhs_19)
                    T.gemm(cast_lhs_19, input_15, matmul_19, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
                    for i0 in T.Parallel(block_t_q):
                        scale_old_state_pass0_m[i0] = state_pass0_m[i0]
                        state_pass0_m[i0] = T.max(scale_old_state_pass0_m[i0], red_max_9[i0])
                        scale_old_state_pass0_m[i0] = T.exp2((scale_old_state_pass0_m[i0] - state_pass0_m[i0]) * scale)
                        scale_tile_state_pass0_m[i0] = T.exp2((red_max_9[i0] - state_pass0_m[i0]) * scale)
                        state_pass0_l[i0] = state_pass0_l[i0] * scale_old_state_pass0_m[i0] + red_sum_22[i0] * scale_tile_state_pass0_m[i0]
                    for i0, i1 in T.Parallel(block_t_q, d_h):
                        state_pass0_o[i0, i1] = state_pass0_o[i0, i1] * scale_old_state_pass0_m[i0] + matmul_19[i0, i1] * scale_tile_state_pass0_m[i0]
                for i0, i1 in T.Parallel(block_t_q, d_h):
                    state_pass0_o[i0, i1] = state_pass0_o[i0, i1] / state_pass0_l[i0]
                T.copy(state_pass0_o, O[gy, gx * block_t_q, 0])
        return main

    return attention_c2349_jit()

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
