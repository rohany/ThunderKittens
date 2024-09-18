import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import time
import thunderkittens as tk

def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    flop = flop / 1e12
    time = time / 1e6
    return flop / time

def benchmark_attention(configurations):
    results = {
        'fwd': defaultdict(list),
        'bwd': defaultdict(list)
    }
    
    for B, H, N, D, causal in configurations:
        print("=" * 60)
        print(f"Timing forward and backward pass for B={B}, H={H}, N={N}, D={D}, causal={causal}")

        q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda')

        o = torch.zeros_like(q)
        grad_output = torch.randn_like(q, requires_grad=False)

        qg = torch.zeros_like(q, requires_grad=False, dtype=torch.float32)
        kg = torch.zeros_like(k, requires_grad=False, dtype=torch.float32)
        vg = torch.zeros_like(v, requires_grad=False, dtype=torch.float32)

        l_vec = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 1, device=q.device, dtype=torch.float32)
        d_vec = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 1, device=q.device, dtype=torch.float32)
        
        # Prepare for timing forward pass
        start_events_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        end_events_fwd = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        
        # Warmup for forward pass
        for _ in range(10):
            tk.mha(q, k, v, o, l_vec, causal)

        # Time the forward pass
        for i in range(100):
            o.zero_()
            
            torch.cuda.synchronize()
            start_events_fwd[i].record()
            
            tk.mha(q, k, v, o, l_vec, causal)
            
            torch.cuda.synchronize()
            end_events_fwd[i].record()

        torch.cuda.synchronize()
        times_fwd = [s.elapsed_time(e) for s, e in zip(start_events_fwd, end_events_fwd)]
        time_us_fwd = np.mean(times_fwd) * 1000

        tflops_fwd = efficiency(flops(B, N, H, D, causal, 'fwd'), time_us_fwd)
        results['fwd'][(D, causal)].append((N, tflops_fwd))

        print(f"Average time for forward pass in us: {time_us_fwd:.2f}")
        print(f"Average efficiency for forward pass in TFLOPS: {tflops_fwd}")
        print("-" * 60)
        
        # wait for GPU to reset
        torch.cuda.synchronize()
        
        # clear cache
        torch.cuda.empty_cache()
    
    return results

def plot_results(results):
    os.makedirs('benchmark_results', exist_ok=True)
    for mode in ['fwd', 'bwd']:
        for (D, causal), values in results[mode].items():
            seq_lens = [x[0] for x in values]
            tflops = [x[1] for x in values]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(seq_lens)), tflops, tick_label=seq_lens)
            plt.xlabel('Sequence Length')
            plt.ylabel('TFLOPS')
            plt.title(f'{mode.upper()} Pass - Head Dim: {D}, Causal: {causal}')
            plt.grid(True)

            # Adding the numerical y value on top of each bar
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

            filename = f'benchmark_results/{mode}_D{D}_causal{causal}.png'
            plt.savefig(filename)
            plt.close()

# Example list of configurations to test
def weija(x):
    val = x + 191
    val -= (val % 192)
    return val

configurations = [
    (4, 32, weija(2048),    128, False),
    (4, 32, weija(4096),    128, False),
    (4, 32, weija(8192),    128, False),
    (4, 32, weija(16384),    128, False),
    # (4, 32, 768,    128, False),
    # (4, 32, 768*2,  128, False),
    # (4, 32, 768*4,  128, False),
    # (4, 32, 768*8,  128, False),
    # (4, 32, 768*16, 128, False),
    # (4, 32, 768*22, 128, False),
]

results = benchmark_attention(configurations)
plot_results(results)
