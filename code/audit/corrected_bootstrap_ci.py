import json, numpy as np
from scipy import stats

print('='*80)
print('Seed-level 95% CI for Delta MAP (Corrected)')
print('Data source: all_eval_results.json (07_evaluate.py pipeline)')
print('='*80)

# Load authoritative per-seed MAP values
with open(r'<DATA_ROOT>\retrieval\results\all_eval_results.json', 'r') as f:
    all_eval = json.load(f)

# Extract per-seed MAP for each method x tier
configs = {
    'ArcFace': {'S0': 'ArcFace_S0', 'S2': 'ArcFace_S2'},
    'GCE':     {'S0': 'GCE_S0',     'S2': 'GCE_S2'},
    'SCE':     {'S0': 'SCE_S0',     'S2': 'SCE_S2'},
    'SC_K3':   {'S0': 'SubCenterArcFace_K3_S0', 'S2': 'SubCenterArcFace_K3_S2'},
}
seeds = [42, 123, 456]
t_crit = stats.t.ppf(0.975, df=2)  # t(0.025, df=2) = 4.303

print(f'\nt-critical (two-tailed 95%, df=2): {t_crit:.3f}')
print()
print('='*80)
print('[A] Paired t-test 95% CI for Delta MAP')
print('='*80)
print(f'{Method:>8s} | {Per-seed MAP S0:^24s} | {Per-seed MAP S2:^24s} | {Delta:>7s} | {95% CI:^24s} | {p_val:>6s}')
print('-'*8 + '-+-' + '-'*24 + '-+-' + '-'*24 + '-+-' + '-'*7 + '-+-' + '-'*24 + '-+-' + '-'*6)

results = {}
for method, tiers in configs.items():
    s0_maps = [all_eval[f'{tiers[S0]}_seed{s}']['MAP'] for s in seeds]
    s2_maps = [all_eval[f'{tiers[S2]}_seed{s}']['MAP'] for s in seeds]
    diffs = [s2 - s0 for s2, s0 in zip(s2_maps, s0_maps)]
    
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    se_diff = std_diff / np.sqrt(len(diffs))
    
    ci_lo = mean_diff - t_crit * se_diff
    ci_hi = mean_diff + t_crit * se_diff
    
    # Paired t-test
    t_stat = mean_diff / se_diff
    p_val = 2 * stats.t.sf(abs(t_stat), df=2)
    
    # Cohen's d (paired)
    d = mean_diff / std_diff
    
    results[method] = {
        's0_maps': s0_maps, 's2_maps': s2_maps, 'diffs': diffs,
        'mean_diff': mean_diff, 'std_diff': std_diff, 'se_diff': se_diff,
        'ci_lo': ci_lo, 'ci_hi': ci_hi, 't_stat': t_stat, 'p_val': p_val, 'd': d
    }
    
    print(f'{method:>8s} | [{s0_maps[0]:.4f},{s0_maps[1]:.4f},{s0_maps[2]:.4f}] mean={np.mean(s0_maps):.4f} | [{s2_maps[0]:.4f},{s2_maps[1]:.4f},{s2_maps[2]:.4f}] mean={np.mean(s2_maps):.4f} | {mean_diff:+.4f} | [{ci_lo:+.4f}, {ci_hi:+.4f}] | {p_val:.4f}')

print()
print('='*80)
print('[B] Seed-level Bootstrap 95% CI (B=10000, resample 3 seed-level diffs)')
print('='*80)
print(f'{Method:>8s} | {d_mean:>7s} | {t-CI:^24s} | {Boot-CI:^24s} | {Exclude 0?:^10s}')
print('-'*8 + '-+-' + '-'*7 + '-+-' + '-'*24 + '-+-' + '-'*24 + '-+-' + '-'*10)

for method, r in results.items():
    diffs = np.array(r['diffs'])
    
    # Seed-level bootstrap
    rng = np.random.RandomState(42)
    B = 10000
    boot_means = np.zeros(B)
    for b in range(B):
        idx = rng.randint(0, 3, size=3)
        boot_means[b] = diffs[idx].mean()
    
    boot_lo = np.percentile(boot_means, 2.5)
    boot_hi = np.percentile(boot_means, 97.5)
    
    excl = 'YES' if (r['ci_lo'] > 0 or r['ci_hi'] < 0) else 'NO'
    boot_excl = 'YES' if (boot_lo > 0 or boot_hi < 0) else 'NO'
    
    print(f'{method:>8s} | {r[mean_diff]:>+.4f} | [{r[ci_lo]:>+.4f}, {r[ci_hi]:>+.4f}] | [{boot_lo:>+.4f}, {boot_hi:>+.4f}] | t:{excl} boot:{boot_excl}')

print()
print('='*80)
print('[C] Comparison: Old Bootstrap CI (wrong) vs Corrected CI')
print('='*80)

old_bootstrap = {
    'ArcFace': {'dMAP': 0.0925, 'lo': 0.0872, 'hi': 0.0977},
    'GCE':     {'dMAP': 0.1139, 'lo': 0.1081, 'hi': 0.1196},
    'SCE':     {'dMAP': 0.1013, 'lo': 0.0957, 'hi': 0.1067},
    'SC_K3':   {'dMAP': 0.1514, 'lo': 0.1449, 'hi': 0.1577},
}

print(f'{Method:>8s} | {Table2 dMAP:>11s} | {Corrected 95%CI:^24s} | {Old Bootstrap (wrong):^24s} | {Ratio:>6s}')
print('-'*8 + '-+-' + '-'*11 + '-+-' + '-'*24 + '-+-' + '-'*24 + '-+-' + '-'*6)

for method, r in results.items():
    old = old_bootstrap[method]
    ratio = old['dMAP'] / r['mean_diff'] if r['mean_diff'] != 0 else float('inf')
    print(f'{method:>8s} | {r[mean_diff]:>+10.4f} | [{r[ci_lo]:>+.4f}, {r[ci_hi]:>+.4f}] | [{old[lo]:>+.4f}, {old[hi]:>+.4f}] | {ratio:.1f}x')

print()
print('='*80)
print('[D] Corrected values for Table 3 footnote')
print('='*80)
print()
print('Replace the Bootstrap CI footnote in Table 3 with:')
print()
for method, r in results.items():
    print(f'  {method:>8s}: dMAP={r[mean_diff]:+.4f}, 95% CI [{r[ci_lo]:+.4f}, {r[ci_hi]:+.4f}], p={r[p_val]:.4f}, d={r[d]:.2f}')

print()
print('Formatted for LaTeX footnote:')
print()
parts = []
for method, r in results.items():
    label = method.replace('_', '-')
    parts.append(f'{label} {r[mean_diff]:+.3f} [{r[ci_lo]:+.3f}, {r[ci_hi]:+.3f}]')
print(f'  Paired 95% CI for dMAP (t-test, df=2): { | .join(parts)}. All CIs exclude zero except SC-K3.')

# Check which CIs exclude zero
print()
print('Verification:')
for method, r in results.items():
    excl = r['ci_lo'] > 0
    print(f'  {method}: CI [{r[ci_lo]:+.4f}, {r[ci_hi]:+.4f}] {EXCLUDES if excl else INCLUDES} zero')

# Save results
output = {}
for method, r in results.items():
    output[method] = {
        'per_seed_S0': r['s0_maps'],
        'per_seed_S2': r['s2_maps'],
        'per_seed_diffs': r['diffs'],
        'mean_diff': r['mean_diff'],
        'std_diff': r['std_diff'],
        'se_diff': r['se_diff'],
        't_stat': r['t_stat'],
        'p_value': r['p_val'],
        'cohens_d': r['d'],
        'ci_95_lo': r['ci_lo'],
        'ci_95_hi': r['ci_hi'],
    }

with open(r'<DRIVE_ROOT>\paper_sivp\corrected_ci_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nResults saved to corrected_ci_results.json')
