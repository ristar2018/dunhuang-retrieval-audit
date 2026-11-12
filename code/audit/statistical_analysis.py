"""
Statistical-significance analysis script
- Bonferroni correction plus FDR (Benjamini-Hochberg) correction
-  paired t-test (pairing on the same seed)
-  independent t-test (comparison across methods)
-  effect size Cohen's d
"""
import json, re, sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

BASE_DIR = Path(r"<DATA_ROOT>\retrieval")
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "experiment_records"

# ======================== data loading ========================
def load_eval_results():
    with open(RESULTS_DIR / "all_eval_results.json", 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_exp_name(exp_name):
    """Parse an experiment name and return (method, dataset, seed)"""
    # SubCenterArcFace_K3_S2_seed42
    # ArcFace_S1_prime_seed123
    # GCE_S0_seed42
    m = re.match(r'(SubCenterArcFace_K\d+|[A-Za-z]+)_(.+?)_seed(\d+)', exp_name)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None

def build_data(all_eval):
    """Build the nested method -> dataset -> seed -> metrics dictionary"""
    data = defaultdict(lambda: defaultdict(dict))
    for exp_name, metrics in all_eval.items():
        method, dataset, seed = parse_exp_name(exp_name)
        if method and dataset:
            data[method][dataset][seed] = metrics
    return data

# ======================== statistical tests ========================
def bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni correction"""
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    return corrected

def fdr_bh_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction"""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    cummin = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        adjusted = p * n / rank
        cummin = min(adjusted, cummin)
        corrected[orig_idx] = min(cummin, 1.0)
    return corrected

def cohens_d_paired(x, y):
    """Paired-sample Cohen's d"""
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0

def cohens_d_independent(x, y):
    """Independent-sample Cohen's d"""
    nx, ny = len(x), len(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx-1)*var_x + (ny-1)*var_y) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0.0

# ======================== analysis 1: noise-gradient effect ========================
def analysis_gradient_effect(data):
    """
    H1: under moderate noise (S2) the MAP is significantly higher than under clean labels (S0)
    for each method, a paired t-test compares S2 with S0 (paired by seed)
    """
    print("=" * 70)
    print("分析1: 噪声梯度效应 (S2 vs S0)")
    print("H0: MAP_S2 = MAP_S0, H1: MAP_S2 > MAP_S0 (单侧)")
    print("=" * 70)
    
    # Phase-2 methods (S0 and S2 both available)
    phase2_methods = ['ArcFace', 'GCE', 'SCE', 'SubCenterArcFace_K3']
    results = []
    
    for method in phase2_methods:
        if 'S0' not in data[method] or 'S2' not in data[method]:
            continue
        seeds_common = sorted(set(data[method]['S0'].keys()) & set(data[method]['S2'].keys()))
        if len(seeds_common) < 3:
            continue
        
        map_s0 = [data[method]['S0'][s]['MAP'] for s in seeds_common]
        map_s2 = [data[method]['S2'][s]['MAP'] for s in seeds_common]
        
        t_stat, p_val = stats.ttest_rel(map_s2, map_s0)
        # one-sided: p/2 if t > 0, else 1 - p/2
        if t_stat > 0:
            p_one_sided = p_val / 2
        else:
            p_one_sided = 1 - p_val / 2
        
        d = cohens_d_paired(map_s2, map_s0)
        
        results.append({
            'method': method,
            'mean_s0': np.mean(map_s0),
            'mean_s2': np.mean(map_s2),
            'improvement': (np.mean(map_s2) - np.mean(map_s0)) / np.mean(map_s0) * 100,
            't_stat': t_stat,
            'p_two_sided': p_val,
            'p_one_sided': p_one_sided,
            'cohens_d': d,
            'seeds': seeds_common,
        })
    
    # multiple-comparison correction
    p_vals = [r['p_one_sided'] for r in results]
    bonf_p = bonferroni_correction(p_vals)
    fdr_p = fdr_bh_correction(p_vals)
    
    print(f"\n{'方法':<30} {'S0 MAP':>8} {'S2 MAP':>8} {'提升%':>7} {'t':>7} {'p(单侧)':>9} {'p_Bonf':>8} {'p_FDR':>8} {'Cohen_d':>8} {'显著':>4}")
    print("-" * 110)
    
    for i, r in enumerate(results):
        sig = ""
        if bonf_p[i] < 0.05:
            sig = "***"
        elif fdr_p[i] < 0.05:
            sig = "**"
        elif r['p_one_sided'] < 0.05:
            sig = "*"
        print(f"{r['method']:<30} {r['mean_s0']:>8.4f} {r['mean_s2']:>8.4f} {r['improvement']:>6.1f}% {r['t_stat']:>7.3f} {r['p_one_sided']:>9.4f} {bonf_p[i]:>8.4f} {fdr_p[i]:>8.4f} {r['cohens_d']:>8.3f} {sig:>4}")
    
    print(f"\n显著性标记: *** p<0.05(Bonferroni) | ** p<0.05(FDR) | * p<0.05(未校正)")
    return results

# ======================== analysis 2: comparison across methods ========================
def analysis_method_comparison(data):
    """
    H1: at tier S2 the MAP of GCE is significantly higher than that of the other methods
    paired t-test (paired by seed)
    """
    print("\n" + "=" * 70)
    print("分析2: S2层方法间比较 (GCE vs 其他)")
    print("=" * 70)
    
    methods = ['ArcFace', 'GCE', 'SCE', 'SubCenterArcFace_K3']
    seeds_all = [42, 123, 456]
    
    # collect MAP at S2 for each method
    method_maps = {}
    for m in methods:
        if 'S2' in data[m]:
            maps = []
            for s in seeds_all:
                if s in data[m]['S2']:
                    maps.append(data[m]['S2'][s]['MAP'])
            if len(maps) == 3:
                method_maps[m] = maps
    
    results = []
    comparisons = []
    
    for other in methods:
        if other == 'GCE' or other not in method_maps:
            continue
        gce_maps = method_maps['GCE']
        other_maps = method_maps[other]
        
        t_stat, p_val = stats.ttest_rel(gce_maps, other_maps)
        if t_stat > 0:
            p_one = p_val / 2
        else:
            p_one = 1 - p_val / 2
        d = cohens_d_paired(gce_maps, other_maps)
        
        comparisons.append(f"GCE vs {other}")
        results.append({
            'comparison': f"GCE vs {other}",
            'gce_mean': np.mean(gce_maps),
            'other_mean': np.mean(other_maps),
            'diff': np.mean(gce_maps) - np.mean(other_maps),
            't_stat': t_stat,
            'p_one_sided': p_one,
            'cohens_d': d,
        })
    
    p_vals = [r['p_one_sided'] for r in results]
    bonf_p = bonferroni_correction(p_vals)
    fdr_p = fdr_bh_correction(p_vals)
    
    print(f"\n{'比较':<30} {'GCE MAP':>9} {'对手MAP':>9} {'差值':>7} {'t':>7} {'p(单侧)':>9} {'p_Bonf':>8} {'p_FDR':>8} {'Cohen_d':>8} {'显著':>4}")
    print("-" * 110)
    
    for i, r in enumerate(results):
        sig = ""
        if bonf_p[i] < 0.05:
            sig = "***"
        elif fdr_p[i] < 0.05:
            sig = "**"
        elif r['p_one_sided'] < 0.05:
            sig = "*"
        print(f"{r['comparison']:<30} {r['gce_mean']:>9.4f} {r['other_mean']:>9.4f} {r['diff']:>+7.4f} {r['t_stat']:>7.3f} {r['p_one_sided']:>9.4f} {bonf_p[i]:>8.4f} {fdr_p[i]:>8.4f} {r['cohens_d']:>8.3f} {sig:>4}")
    
    print(f"\n显著性标记: *** p<0.05(Bonferroni) | ** p<0.05(FDR) | * p<0.05(未校正)")
    return results

# ======================== analysis 3: effect of K ========================
def analysis_kvalue_effect(data):
    """
    H1: at tier S0, K=1 is significantly better than K=3 and K=5
    """
    print("\n" + "=" * 70)
    print("分析3: K值效应 (Phase 3)")
    print("=" * 70)
    
    methods_k = ['SubCenterArcFace_K1', 'SubCenterArcFace_K3', 'SubCenterArcFace_K5']
    seeds_all = [42, 123, 456]
    
    # MAP for each K on each dataset
    print("\n(a) K值 × 数据集 MAP")
    print(f"\n{'K值':<30} {'S0':>8} {'S1':>8} {'S2':>8} {'S3':>8}")
    print("-" * 70)
    
    for m in methods_k:
        if m not in data:
            continue
        row = [m.replace('SubCenterArcFace_', '')]
        for ds in ['S0', 'S1', 'S2', 'S3']:
            if ds in data[m]:
                maps = [data[m][ds][s]['MAP'] for s in seeds_all if s in data[m][ds]]
                row.append(f"{np.mean(maps):.3f}±{np.std(maps):.3f}")
            else:
                row.append('-')
        print(f"{row[0]:<30} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8}")
    
    # K=1 vs K=3, K=1 vs K=5 at S0
    print("\n(b) K=1 vs K=3/K=5 at S0 (配对t-test)")
    comparisons = []
    results = []
    
    for other_k in ['SubCenterArcFace_K3', 'SubCenterArcFace_K5']:
        if other_k not in data or 'SubCenterArcFace_K1' not in data:
            continue
        if 'S0' not in data['SubCenterArcFace_K1'] or 'S0' not in data[other_k]:
            continue
        
        maps_k1 = [data['SubCenterArcFace_K1']['S0'][s]['MAP'] for s in seeds_all if s in data['SubCenterArcFace_K1']['S0']]
        maps_other = [data[other_k]['S0'][s]['MAP'] for s in seeds_all if s in data[other_k]['S0']]
        
        if len(maps_k1) == 3 and len(maps_other) == 3:
            t_stat, p_val = stats.ttest_rel(maps_k1, maps_other)
            if t_stat > 0:
                p_one = p_val / 2
            else:
                p_one = 1 - p_val / 2
            d = cohens_d_paired(maps_k1, maps_other)
            
            k_label = other_k.replace('SubCenterArcFace_', '')
            comparisons.append(f"K=1 vs {k_label}")
            results.append({
                'comparison': f"K=1 vs {k_label}",
                'k1_mean': np.mean(maps_k1),
                'other_mean': np.mean(maps_other),
                't_stat': t_stat,
                'p_one_sided': p_one,
                'cohens_d': d,
            })
    
    if results:
        p_vals = [r['p_one_sided'] for r in results]
        bonf_p = bonferroni_correction(p_vals)
        fdr_p = fdr_bh_correction(p_vals)
        
        print(f"\n{'比较':<20} {'K=1 MAP':>9} {'对手MAP':>9} {'t':>7} {'p(单侧)':>9} {'p_Bonf':>8} {'p_FDR':>8} {'Cohen_d':>8} {'显著':>4}")
        print("-" * 90)
        for i, r in enumerate(results):
            sig = "***" if bonf_p[i] < 0.05 else ("**" if fdr_p[i] < 0.05 else ("*" if r['p_one_sided'] < 0.05 else ""))
            print(f"{r['comparison']:<20} {r['k1_mean']:>9.4f} {r['other_mean']:>9.4f} {r['t_stat']:>7.3f} {r['p_one_sided']:>9.4f} {bonf_p[i]:>8.4f} {fdr_p[i]:>8.4f} {r['cohens_d']:>8.3f} {sig:>4}")
    
    # is the noise-gradient effect significant at K = 1
    print("\n(c) K=1的噪声梯度效应 (S2 vs S0)")
    if 'SubCenterArcFace_K1' in data:
        m = 'SubCenterArcFace_K1'
        maps_s0 = [data[m]['S0'][s]['MAP'] for s in seeds_all if s in data[m]['S0']]
        maps_s2 = [data[m]['S2'][s]['MAP'] for s in seeds_all if s in data[m]['S2']]
        if len(maps_s0) == 3 and len(maps_s2) == 3:
            t_stat, p_val = stats.ttest_rel(maps_s2, maps_s0)
            p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
            d = cohens_d_paired(maps_s2, maps_s0)
            print(f"  S0 MAP = {np.mean(maps_s0):.4f}±{np.std(maps_s0):.4f}")
            print(f"  S2 MAP = {np.mean(maps_s2):.4f}±{np.std(maps_s2):.4f}")
            print(f"  提升 = {(np.mean(maps_s2)-np.mean(maps_s0))/np.mean(maps_s0)*100:.1f}%")
            print(f"  t = {t_stat:.3f}, p(单侧) = {p_one:.4f}, Cohen's d = {d:.3f}")
            print(f"  结论: {'显著' if p_one < 0.05 else '不显著'}")

# ======================== analysis 4: hyperparameters vs sub-centres ========================
def analysis_hyperparam_vs_subcenter(data):
    """
    ArcFace(scale=30) vs SubCenterK1(scale=64) at S0
    verify the effect of the hyperparameter choice
    """
    print("\n" + "=" * 70)
    print("分析4: 超参数效应 vs 子中心效应")
    print("ArcFace(scale=30,margin=0.5) vs SubCenterK1(scale=64,margin=28.6)")
    print("=" * 70)
    
    seeds_all = [42, 123, 456]
    
    if 'ArcFace' not in data or 'SubCenterArcFace_K1' not in data:
        print("数据不足，跳过")
        return
    
    for ds in ['S0', 'S2']:
        maps_af = [data['ArcFace'][ds][s]['MAP'] for s in seeds_all if s in data['ArcFace'].get(ds, {})]
        maps_k1 = [data['SubCenterArcFace_K1'][ds][s]['MAP'] for s in seeds_all if s in data['SubCenterArcFace_K1'].get(ds, {})]
        
        if len(maps_af) == 3 and len(maps_k1) == 3:
            t_stat, p_val = stats.ttest_rel(maps_k1, maps_af)
            d = cohens_d_paired(maps_k1, maps_af)
            print(f"\n  [{ds}] ArcFace: {np.mean(maps_af):.4f}±{np.std(maps_af):.4f}")
            print(f"  [{ds}] K1:      {np.mean(maps_k1):.4f}±{np.std(maps_k1):.4f}")
            print(f"  差值: {np.mean(maps_k1)-np.mean(maps_af):+.4f}")
            print(f"  t = {t_stat:.3f}, p = {p_val:.4f}, Cohen's d = {d:.3f}")
            print(f"  结论: {'K1显著优于ArcFace(超参效应)' if t_stat > 0 and p_val < 0.05 else '差异不显著' if p_val >= 0.05 else 'ArcFace优于K1'}")

# ======================== analysis 5: pooled effect sizes ========================
def analysis_effect_sizes(data):
    """Summarise the effect sizes of every key comparison"""
    print("\n" + "=" * 70)
    print("分析5: 效应量汇总 (Cohen's d)")
    print("=" * 70)
    
    seeds_all = [42, 123, 456]
    
    print(f"\n{'比较':<35} {'Cohen_d':>9} {'效应大小':>10}")
    print("-" * 60)
    
    effect_size_results = []
    
    # gradient effect (S2 vs S0) for each method
    for method in ['ArcFace', 'GCE', 'SCE', 'SubCenterArcFace_K3', 'SubCenterArcFace_K1']:
        if method not in data:
            continue
        if 'S0' not in data[method] or 'S2' not in data[method]:
            continue
        maps_s0 = [data[method]['S0'][s]['MAP'] for s in seeds_all if s in data[method]['S0']]
        maps_s2 = [data[method]['S2'][s]['MAP'] for s in seeds_all if s in data[method]['S2']]
        if len(maps_s0) == 3 and len(maps_s2) == 3:
            d = cohens_d_paired(maps_s2, maps_s0)
            label = f"{method} S2 vs S0"
            size = "大" if abs(d) >= 0.8 else ("中" if abs(d) >= 0.5 else "小")
            print(f"{label:<35} {d:>9.3f} {size:>10}")
            effect_size_results.append({'comparison': label, 'd': d, 'size': size})
    
    # effect of K
    for other in ['SubCenterArcFace_K3', 'SubCenterArcFace_K5']:
        if 'SubCenterArcFace_K1' not in data or other not in data:
            continue
        maps_k1 = [data['SubCenterArcFace_K1']['S0'][s]['MAP'] for s in seeds_all if s in data['SubCenterArcFace_K1']['S0']]
        maps_other = [data[other]['S0'][s]['MAP'] for s in seeds_all if s in data[other]['S0']]
        if len(maps_k1) == 3 and len(maps_other) == 3:
            d = cohens_d_paired(maps_k1, maps_other)
            k_label = other.replace('SubCenterArcFace_', '')
            label = f"K=1 vs {k_label} (S0)"
            size = "大" if abs(d) >= 0.8 else ("中" if abs(d) >= 0.5 else "小")
            print(f"{label:<35} {d:>9.3f} {size:>10}")
            effect_size_results.append({'comparison': label, 'd': d, 'size': size})
    
    return effect_size_results

# ======================== main flow ========================
def main():
    print("统计显著性分析报告")
    print(f"数据源: {RESULTS_DIR / 'all_eval_results.json'}")
    print(f"分析时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_eval = load_eval_results()
    data = build_data(all_eval)
    
    # 5 analyses
    r1 = analysis_gradient_effect(data)
    r2 = analysis_method_comparison(data)
    analysis_kvalue_effect(data)
    analysis_hyperparam_vs_subcenter(data)
    r5 = analysis_effect_sizes(data)
    
    # save the results as JSON
    summary = {
        'analysis_time': __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_experiments': len(all_eval),
        'gradient_effect': [{
            'method': r['method'],
            'mean_s0': round(r['mean_s0'], 4),
            'mean_s2': round(r['mean_s2'], 4),
            'improvement_pct': round(r['improvement'], 1),
            'p_one_sided': round(r['p_one_sided'], 4),
            'cohens_d': round(r['cohens_d'], 3),
        } for r in r1],
        'method_comparison_s2': [{
            'comparison': r['comparison'],
            'diff': round(r['diff'], 4),
            'p_one_sided': round(r['p_one_sided'], 4),
            'cohens_d': round(r['cohens_d'], 3),
        } for r in r2],
    }
    
    out_path = OUTPUT_DIR / "statistical_analysis_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n统计结果已保存: {out_path}")

if __name__ == '__main__':
    main()
