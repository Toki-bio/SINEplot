#!/usr/bin/env python3
"""
SINE MSA-PCA Discriminator

Refines SINEderella subfamily assignments for closely related subfamilies
using sequence-aware PCA on one-hot encoded alignments.

SINEderella already does the heavy lifting (genome search, extraction, bitscore
assignment across 50+ subfamilies).  This tool takes SINEderella's output and
applies MSA-PCA discrimination to separate copies that bitscores cannot cleanly
resolve — e.g. sq2s / sq2l / sq2m.

Primary inputs (from SINEderella):
  --assignment    assignment_full.tsv from SINEderella step2
  --extracted     extracted.fasta from SINEderella step1
  --consensuses   Consensus FASTA for the target subfamilies only
  --subfamilies   Comma-separated subfamily names to discriminate (e.g. sq2s,sq2l,sq2m)

Alternative input (standalone):
  --scores        ssearch36 -m8 output (if not using SINEderella assignment)

Alignment:
  --aligned       Pre-computed MAFFT alignment (skips MAFFT step)
                  If omitted, runs MAFFT --add --keeplength automatically

Discrimination method:
  1. Filter SINEderella assignments for target subfamilies only
  2. Extract matching copy sequences
  3. Profile-align copies against pre-aligned consensuses (MAFFT --add --keeplength)
  4. One-hot encode alignment → PCA (captures SNPs + diagnostic indels)
  5. Mahalanobis distance to each subfamily cluster
  6. Three-layer decision: ratio test + outlier detection + split-signal (chimera) test

Outputs (to --outdir):
  <subfamily>_clean.fa       — confidently assigned copies
  grey_zone.fa               — ambiguous/chimeric/outlier copies
  discrimination_report.tsv  — per-copy distances, ratios, categories
"""

import argparse
import os
import sys
import random
import subprocess
import tempfile
import numpy as np
from collections import defaultdict


def _mahalanobis(u, v, VI):
    """Mahalanobis distance (pure numpy, replaces scipy)."""
    delta = np.asarray(u) - np.asarray(v)
    return np.sqrt(np.dot(np.dot(delta, VI), delta))


def _pca_via_svd(X, n_components):
    """PCA via SVD on centred data (pure numpy, replaces sklearn).

    Returns:
      coords:  (n_samples, n_components) projected coordinates
      explained_variance_ratio: array of length n_components
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = U[:, :n_components] * S[:n_components]
    total_var = (S ** 2).sum()
    evr = (S[:n_components] ** 2) / total_var if total_var > 0 else np.zeros(n_components)
    return coords, evr


def parse_fasta(filepath):
    """Parse FASTA file into dict of name -> sequence."""
    sequences = {}
    current_name = None
    current_seq = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name is not None:
                    sequences[current_name] = ''.join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            elif current_name is not None:
                current_seq.append(line)
    if current_name is not None:
        sequences[current_name] = ''.join(current_seq)
    return sequences


def parse_scores(filepath):
    """Parse ssearch36 -m8 output. Returns dict of (query, subject) -> max_bitscore."""
    scores = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) >= 12:
                try:
                    q, s = fields[0], fields[1]
                    bs = float(fields[11])
                    key = (q, s)
                    if key not in scores or bs > scores[key]:
                        scores[key] = bs
                except (ValueError, IndexError):
                    continue
    return scores


def parse_assignment_tsv(filepath, target_subfamilies):
    """Parse SINEderella assignment_full.tsv, filter for target subfamilies.

    assignment_full.tsv columns: Seq, Subfamily, Bitscore, Votes, Status, Threshold

    Returns:
      assignments: dict copy_name -> subfamily (only for target subfamilies)
      bitscores:   dict copy_name -> bitscore (from SINEderella)
    """
    assignments = {}
    bitscores = {}
    target_set = set(target_subfamilies)
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Seq') or line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) >= 3:
                seq_name = fields[0]
                subfamily = fields[1]
                try:
                    bitscore = float(fields[2])
                except ValueError:
                    bitscore = 0
                if subfamily in target_set:
                    assignments[seq_name] = subfamily
                    bitscores[seq_name] = bitscore
    return assignments, bitscores


def extract_target_copies(extracted_fasta, copy_names):
    """Extract sequences for target copy names from SINEderella extracted.fasta.

    Returns dict of name -> sequence (unaligned).
    """
    all_seqs = parse_fasta(extracted_fasta)
    result = {}
    for name in copy_names:
        if name in all_seqs:
            result[name] = all_seqs[name]
    return result


def pairwise_align_copies(cons_aln_path, copy_names, copy_seqs, outdir, jobs=4):
    """Align all copies to consensus profile in a single MAFFT batch call.

    Uses mafft --add <all_copies.fa> --keeplength <cons_aln> --thread N
    which is orders of magnitude faster than one subprocess per copy.

    Returns dict of copy_name -> aligned_sequence (in consensus coordinate frame).
    """
    os.makedirs(outdir, exist_ok=True)

    # Write all copies to one FASTA file
    # Use indexed headers to avoid shell/filesystem issues; map back after
    queries_fa = os.path.join(outdir, "_queries.fa")
    idx_to_name = {}
    with open(queries_fa, 'w') as f:
        for i, name in enumerate(copy_names):
            if name in copy_seqs:
                tag = f"COPY{i}"
                idx_to_name[tag] = name
                f.write(f">{tag}\n{copy_seqs[name]}\n")

    total = len(idx_to_name)
    print(f"    Batch-aligning {total} copies (1 MAFFT call, {jobs} threads)...", flush=True)

    result = subprocess.run(
        ['mafft', '--add', queries_fa, '--keeplength',
         '--thread', str(jobs), cons_aln_path],
        capture_output=True, timeout=3600
    )
    if result.returncode != 0:
        err = result.stderr.decode('utf-8', errors='replace')[:500]
        print(f"  ERROR: mafft batch failed: {err}")
        return {}

    # Parse output — keep only the copy sequences (not the consensuses)
    output = result.stdout.decode('utf-8', errors='replace')
    aligned = {}
    current_tag = None
    current_seq = []
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if current_tag is not None and current_tag in idx_to_name:
                aligned[idx_to_name[current_tag]] = ''.join(current_seq)
            current_tag = line[1:].split()[0]
            current_seq = []
        elif current_tag is not None:
            current_seq.append(line)
    if current_tag is not None and current_tag in idx_to_name:
        aligned[idx_to_name[current_tag]] = ''.join(current_seq)

    failed = total - len(aligned)
    print(f"    {len(aligned)}/{total} copies aligned ({failed} failed)", flush=True)
    return aligned


def encode_alignment_onehot(sequences, names):
    """One-hot encode aligned sequences. Returns (n x positions*5) matrix."""
    aln_len = len(sequences[names[0]])
    nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
    n = len(names)
    matrix = np.zeros((n, aln_len * 5), dtype=np.float32)
    for i, name in enumerate(names):
        seq = sequences[name].upper()
        for j, ch in enumerate(seq):
            idx = nuc_map.get(ch)
            if idx is not None:
                matrix[i, j * 5 + idx] = 1.0
    return matrix


def get_bitscore_assignment(scores, subfamilies, copies):
    """Assign each copy to best subfamily by bitscore (SINEderella-style)."""
    assignments = {}
    for copy in copies:
        best_sf = None
        best_score = -1
        for sf in subfamilies:
            s = scores.get((sf, copy), scores.get((copy, sf), 0))
            if s > best_score:
                best_score = s
                best_sf = sf
        assignments[copy] = best_sf
    return assignments


def compute_discrimination(aligned_file, consensus_names, copy_names, n_components=5):
    """Run PCA on one-hot MSA, compute Mahalanobis distances per copy to each subfamily.

    Returns:
      coords_2d: dict name -> (pc1, pc2) for visualization
      distances: dict copy_name -> {subfamily: mahalanobis_distance}
      pca_result: dict with variance_explained, n_components_used
    """
    sequences = parse_fasta(aligned_file)

    all_names = consensus_names + copy_names
    present = [n for n in all_names if n in sequences]
    missing = set(all_names) - set(present)
    if missing:
        print(f"  WARNING: {len(missing)} names not in alignment")

    # One-hot encode
    matrix = encode_alignment_onehot(sequences, present)

    # Remove constant columns
    col_var = matrix.var(axis=0)
    variable_cols = col_var > 0
    matrix_var = matrix[:, variable_cols]
    n_var = matrix_var.shape[1]
    print(f"  Alignment: {len(present)} sequences, {matrix.shape[1]} features -> {n_var} variable")

    # PCA — use more components for distance calculation, show 2 for viz
    n_comp = min(n_components, n_var, len(present) - 1)
    coords, ve = _pca_via_svd(matrix_var, n_comp)
    print(f"  PCA: {n_comp} components, variance explained = {sum(ve):.1%}")
    print(f"    PC1={ve[0]:.1%}, PC2={ve[1]:.1%}" +
          (f", PC3={ve[2]:.1%}" if n_comp > 2 else ""))

    # Build name -> index mapping
    name_idx = {n: i for i, n in enumerate(present)}

    # 2D coords for visualization
    coords_2d = {}
    for name in present:
        i = name_idx[name]
        coords_2d[name] = (coords[i, 0], coords[i, 1])

    # Group copies by bitscore-assigned subfamily for covariance estimation
    # First, we need subfamily centroids from ALL core copies (not just consensus)
    # We'll compute cluster stats from copies assigned to each subfamily

    return coords, coords_2d, present, name_idx, ve, n_comp


def detect_present_subfamilies(coords, name_idx, subfamilies, copy_names,
                               bitscore_assignments, min_copies=10, min_separation=1.0):
    """Detect which subfamilies genuinely exist in this genome.

    Uses cluster separation in PCA space: a subfamily is considered present only
    if its bitscore-assigned copies form a cluster that is geometrically distinct
    from all other subfamily clusters.

    Separation score for a pair (A, B):
        separation = centroid_distance(A, B) / (mean_spread_A + mean_spread_B)

    If separation < min_separation for every other subfamily, the clusters overlap
    too much to be biologically distinct — the minor one is absent.

    Copies are grouped by SINEderella bitscore assignment (not by nearest reference
    in PCA space), so the test is: "do the copies SINEderella called sq2m form a
    distinct cloud from the copies SINEderella called sq2l?"  If not, sq2m doesn't
    exist here — those copies are just edge-members of the sq2l cloud that happened
    to score slightly better against sq2m.

    Also applies a minimum member count: a subfamily with fewer than min_copies
    bitscore-assigned copies in the alignment is considered absent regardless.

    Returns:
      present: list of subfamily names that are genuinely distinct
      absent:  dict of subfamily name -> reason string (for reporting)
    """
    # Group by SINEderella bitscore assignment
    sf_members = defaultdict(list)
    for copy in copy_names:
        if copy not in name_idx:
            continue
        sf = bitscore_assignments.get(copy)
        if sf and sf in subfamilies:
            sf_members[sf].append(copy)

    # Compute per-subfamily centroids and mean intra-cluster spread
    sf_centroid = {}
    sf_spread = {}
    absent = {}

    for sf in subfamilies:
        members = sf_members[sf]
        if len(members) < min_copies:
            absent[sf] = (f"too few bitscore-assigned copies in alignment "
                          f"({len(members)} < {min_copies})")
            continue
        mc = np.array([coords[name_idx[m]] for m in members])
        centroid = mc.mean(axis=0)
        spread = np.mean(np.linalg.norm(mc - centroid, axis=1))
        sf_centroid[sf] = centroid
        sf_spread[sf] = max(spread, 1e-6)

    # For each candidate-present subfamily, check separation from all others
    candidate_sfs = list(sf_centroid.keys())
    # Iterate over a stable snapshot; absent dict may grow inside the loop
    for sf in list(candidate_sfs):
        if sf in absent:
            continue
        others = [o for o in candidate_sfs if o != sf and o not in absent]
        if not others:
            continue
        max_sep = 0.0
        for other in others:
            dist = np.linalg.norm(sf_centroid[sf] - sf_centroid[other])
            sep = dist / (sf_spread[sf] + sf_spread[other])
            if sep > max_sep:
                max_sep = sep

        if max_sep < min_separation:
            # Exclude the smaller of the two overlapping subfamilies
            n_self = len(sf_members[sf])
            closest = max(others, key=lambda o: len(sf_members[o]))
            n_closest = len(sf_members[closest])
            if n_self < n_closest:
                absent[sf] = (f"cluster not distinct from '{closest}' "
                              f"(separation={max_sep:.2f} < {min_separation}), "
                              f"{n_self} copies reassigned")

    present = [sf for sf in subfamilies if sf not in absent]
    return present, absent


def discriminate(coords, name_idx, subfamilies, copies, bitscore_assignments,
                 ratio_threshold=1.5, max_distance_percentile=95,
                 split_signal_threshold=0.4, variance_ratios=None):
    """Classify copies as clean or grey zone using Mahalanobis distance in PCA space.

    Two-layer decision:
      1. Outlier detection (distance beyond cluster percentile)
      2. Mahalanobis ratio test (2nd_closest / closest)

    Split-signal score is computed and reported but NOT used for classification.
    Per-PC voting is too noisy for closely related subfamilies where low-variance
    PCs are dominated by noise rather than subfamily-discriminating signal.

    Parameters:
      ratio_threshold: min ratio of 2nd_closest / closest Mahalanobis distance for "clean"
      max_distance_percentile: copies beyond this percentile of their assigned subfamily
                               are grey-zoned regardless of ratio
      split_signal_threshold: (unused in classification, kept for reporting)
      variance_ratios: array of explained variance ratios per PC (for split score computation)
    """
    # Group copies by bitscore assignment
    sf_members = defaultdict(list)
    for copy in copies:
        if copy in name_idx:
            sf = bitscore_assignments.get(copy)
            if sf:
                sf_members[sf].append(copy)

    # Compute per-subfamily stats in PCA space
    sf_centroids = {}
    sf_cov_inv = {}
    sf_distances_within = {}  # for percentile thresholds

    for sf in subfamilies:
        members = sf_members[sf]
        if len(members) < 3:
            print(f"  WARNING: subfamily '{sf}' has only {len(members)} members, using Euclidean")
            if members:
                member_coords = np.array([coords[name_idx[m]] for m in members])
                sf_centroids[sf] = member_coords.mean(axis=0)
            elif sf in name_idx:
                sf_centroids[sf] = coords[name_idx[sf]]
            else:
                sf_centroids[sf] = np.zeros(coords.shape[1])
            sf_cov_inv[sf] = None
            continue

        member_coords = np.array([coords[name_idx[m]] for m in members])
        centroid = member_coords.mean(axis=0)
        sf_centroids[sf] = centroid

        # Covariance with regularization
        cov = np.cov(member_coords.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        # Regularize: add small identity to prevent singular matrix
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            cov_inv = np.linalg.inv(cov)
            sf_cov_inv[sf] = cov_inv
        except np.linalg.LinAlgError:
            print(f"  WARNING: singular covariance for '{sf}', using Euclidean")
            sf_cov_inv[sf] = None

        # Compute within-cluster distances for percentile threshold
        dists = []
        for m in members:
            d = _mahalanobis(coords[name_idx[m]], centroid, cov_inv)
            dists.append(d)
        sf_distances_within[sf] = dists

    # Compute distance thresholds per subfamily
    sf_max_distance = {}
    for sf in subfamilies:
        if sf in sf_distances_within and sf_distances_within[sf]:
            sf_max_distance[sf] = np.percentile(sf_distances_within[sf], max_distance_percentile)
        else:
            sf_max_distance[sf] = float('inf')

    # Classify each copy
    results = []
    for copy in copies:
        if copy not in name_idx:
            results.append({
                'copy': copy, 'assigned_sf': 'MISSING', 'pca_sf': 'MISSING',
                'category': 'missing', 'closest_dist': float('inf'),
                'second_dist': float('inf'), 'ratio': 0,
                'distances': {sf: float('inf') for sf in subfamilies},
            })
            continue

        # Distance to each subfamily
        dists = {}
        for sf in subfamilies:
            centroid = sf_centroids[sf]
            cov_inv = sf_cov_inv.get(sf)
            if cov_inv is not None:
                dists[sf] = _mahalanobis(coords[name_idx[copy]], centroid, cov_inv)
            else:
                dists[sf] = np.linalg.norm(coords[name_idx[copy]] - centroid)

        sorted_sfs = sorted(dists, key=lambda s: dists[s])
        closest_sf = sorted_sfs[0]
        closest_dist = dists[closest_sf]

        if len(sorted_sfs) > 1:
            second_sf = sorted_sfs[1]
            second_dist = dists[second_sf]
        else:
            second_sf = closest_sf
            second_dist = float('inf')

        ratio = second_dist / closest_dist if closest_dist > 0 else float('inf')
        max_d = sf_max_distance.get(closest_sf, float('inf'))

        # Split-signal test: variance-weighted per-PC dimension voting
        # Each PC's vote is weighted by its explained variance ratio,
        # so high-variance PCs dominate over noisy ones.
        n_dims = coords.shape[1]
        copy_coords = coords[name_idx[copy]]
        dim_votes = {}
        for d in range(n_dims):
            weight = variance_ratios[d] if variance_ratios is not None else 1.0
            best_sf_dim = min(subfamilies,
                              key=lambda sf: abs(copy_coords[d] - sf_centroids[sf][d]))
            dim_votes[best_sf_dim] = dim_votes.get(best_sf_dim, 0) + weight

        # Fraction of variance-weighted votes that disagree with overall closest
        total_weight = sum(variance_ratios[:n_dims]) if variance_ratios is not None else n_dims
        agree_frac = dim_votes.get(closest_sf, 0) / total_weight
        split_score = 1.0 - agree_frac  # 0 = fully consistent, 1 = fully split

        # Decision (two-layer: outlier then ratio)
        if closest_dist > max_d * 2:
            category = 'grey_outlier'
        elif ratio >= ratio_threshold:
            category = 'clean'
        else:
            category = 'grey_ambiguous'

        results.append({
            'copy': copy,
            'assigned_sf': bitscore_assignments.get(copy, 'UNKNOWN'),
            'pca_sf': closest_sf,
            'category': category,
            'closest_dist': closest_dist,
            'second_dist': second_dist,
            'ratio': ratio,
            'split_score': split_score,
            'dim_votes': dim_votes,
            'distances': dists,
        })

    return results


def validate_vs_ground_truth(results, truth_file):
    """Compare discrimination results against ground truth labels."""
    # Parse ground truth
    truth = {}
    with open(truth_file, 'r') as f:
        header = f.readline()  # skip
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) >= 3:
                truth[fields[0]] = {'sf': fields[1], 'category': fields[2],
                                    'detail': fields[3] if len(fields) > 3 else ''}

    print("\n" + "=" * 70)
    print("GROUND TRUTH VALIDATION")
    print("=" * 70)

    # Count by true category
    categories = defaultdict(lambda: {'total': 0, 'clean': 0, 'grey_ambiguous': 0,
                                       'grey_outlier': 0, 'grey_chimeric': 0,
                                       'correct_clean': 0})
    for r in results:
        copy = r['copy']
        if copy not in truth:
            continue
        t = truth[copy]
        cat = t['category']
        categories[cat]['total'] += 1
        categories[cat][r['category']] += 1

        # For clean assignments, check if PCA subfamily matches true subfamily
        if r['category'] == 'clean':
            # For chimeras/mimics, "correct" means grey zone would be better
            if cat in ('core', 'divergent'):
                if r['pca_sf'] == t['sf']:
                    categories[cat]['correct_clean'] += 1
            else:
                # chimera/mimic assigned as clean = false confidence
                pass

    for cat in ['core', 'divergent', 'chimera', 'mimic']:
        if cat not in categories:
            continue
        c = categories[cat]
        total = c['total']
        clean = c['clean']
        grey = c['grey_ambiguous'] + c['grey_outlier'] + c['grey_chimeric']
        print(f"\n  {cat.upper()} copies ({total} total):")
        print(f"    Clean:       {clean:3d} ({clean/total*100:.1f}%)")
        print(f"    Grey zone:   {grey:3d} ({grey/total*100:.1f}%)")
        if c['grey_chimeric'] > 0:
            print(f"      chimeric:  {c['grey_chimeric']:3d}")
        if c['grey_ambiguous'] > 0:
            print(f"      ambiguous: {c['grey_ambiguous']:3d}")
        if c['grey_outlier'] > 0:
            print(f"      outlier:   {c['grey_outlier']:3d}")
        if cat in ('core', 'divergent'):
            correct = c['correct_clean']
            print(f"    Correct clean: {correct}/{clean} "
                  f"({correct/clean*100:.1f}%)" if clean > 0 else "    Correct clean: N/A")
        elif cat in ('chimera', 'mimic'):
            # For these, grey zone is the DESIRED outcome
            print(f"    -> Grey zone is CORRECT for {cat}s: {grey}/{total}")

    # Overall accuracy for core+divergent
    assignable = [r for r in results if r['copy'] in truth
                  and truth[r['copy']]['category'] in ('core', 'divergent')]
    clean_correct = sum(1 for r in assignable
                        if r['category'] == 'clean' and r['pca_sf'] == truth[r['copy']]['sf'])
    clean_wrong = sum(1 for r in assignable
                      if r['category'] == 'clean' and r['pca_sf'] != truth[r['copy']]['sf'])
    grey = sum(1 for r in assignable if r['category'] != 'clean')
    total = len(assignable)

    print(f"\n  ASSIGNABLE (core+divergent, n={total}):")
    print(f"    Correct clean:  {clean_correct} ({clean_correct/total*100:.1f}%)")
    print(f"    Wrong clean:    {clean_wrong} ({clean_wrong/total*100:.1f}%)")
    print(f"    Grey zone:      {grey} ({grey/total*100:.1f}%)")
    if clean_correct + clean_wrong > 0:
        precision = clean_correct / (clean_correct + clean_wrong)
        print(f"    Clean precision: {precision:.1%}")

    # Chimera/mimic detection rate
    tricky = [r for r in results if r['copy'] in truth
              and truth[r['copy']]['category'] in ('chimera', 'mimic')]
    caught = sum(1 for r in tricky if r['category'] != 'clean')
    print(f"\n  TRICKY (chimera+mimic, n={len(tricky)}):")
    print(f"    Caught as grey: {caught}/{len(tricky)} ({caught/len(tricky)*100:.1f}%)")
    print(f"    Escaped as clean: {len(tricky)-caught}/{len(tricky)}")


def write_outputs(results, outdir, aligned_file):
    """Write BED, FASTA, and report files."""
    os.makedirs(outdir, exist_ok=True)

    # Parse aligned sequences for FASTA output
    sequences = parse_fasta(aligned_file)

    # Group by assignment
    clean_by_sf = defaultdict(list)
    grey = []
    for r in results:
        if r['category'] == 'clean':
            clean_by_sf[r['pca_sf']].append(r)
        else:
            grey.append(r)

    # Write per-subfamily clean files
    for sf, members in sorted(clean_by_sf.items()):
        fa_path = os.path.join(outdir, f"{sf}_clean.fa")
        with open(fa_path, 'w', newline='\n') as f:
            for r in members:
                seq = sequences.get(r['copy'], '')
                # Remove gap characters from aligned sequence
                seq_clean = seq.replace('-', '')
                f.write(f">{r['copy']}\n{seq_clean}\n")
        print(f"  {fa_path}: {len(members)} sequences")

    # Write grey zone
    grey_fa = os.path.join(outdir, "grey_zone.fa")
    with open(grey_fa, 'w', newline='\n') as f:
        for r in grey:
            seq = sequences.get(r['copy'], '')
            seq_clean = seq.replace('-', '')
            f.write(f">{r['copy']} pca_closest={r['pca_sf']} "
                    f"ratio={r['ratio']:.2f} cat={r['category']}\n{seq_clean}\n")
    print(f"  {grey_fa}: {len(grey)} sequences")

    # Write full report
    report_path = os.path.join(outdir, "discrimination_report.tsv")
    subfamilies = sorted(set(r['pca_sf'] for r in results if r['pca_sf'] != 'MISSING'))
    with open(report_path, 'w', newline='\n') as f:
        header = ['copy', 'bitscore_sf', 'pca_sf', 'category',
                  'closest_dist', 'second_dist', 'ratio', 'split_score']
        header += [f"dist_{sf}" for sf in subfamilies]
        f.write('\t'.join(header) + '\n')
        for r in results:
            row = [r['copy'], r['assigned_sf'], r['pca_sf'], r['category'],
                   f"{r['closest_dist']:.3f}", f"{r['second_dist']:.3f}",
                   f"{r['ratio']:.3f}", f"{r.get('split_score', 0):.3f}"]
            row += [f"{r['distances'].get(sf, float('inf')):.3f}" for sf in subfamilies]
            f.write('\t'.join(row) + '\n')
    print(f"  {report_path}: {len(results)} rows")

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="SINE MSA-PCA Discriminator — refine SINEderella subfamily assignments",
        epilog=("SINEderella input mode: --assignment + --extracted + --subfamilies + --consensuses\n"
                "Legacy mode: --scores + --aligned + --consensuses"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # SINEderella input mode
    parser.add_argument("--assignment", help="SINEderella assignment_full.tsv")
    parser.add_argument("--extracted", help="SINEderella extracted.fasta (all SINE copies)")
    parser.add_argument("--subfamilies", help="Comma-separated target subfamily names (e.g. sq2s,sq2l,sq2m)")
    # Legacy / shared
    parser.add_argument("--scores", help="ssearch36 -m8 output (legacy mode)")
    parser.add_argument("--aligned", help="Pre-aligned FASTA (skip MAFFT if provided)")
    parser.add_argument("--consensuses", required=True, help="Consensus FASTA (subfamily names)")
    parser.add_argument("--ground-truth", help="Ground truth TSV for validation")
    parser.add_argument("--outdir", default="discrimination_output", help="Output directory")
    parser.add_argument("--n-components", type=int, default=20,
                        help="Number of PCA components for distance calculation (default: 20)")
    parser.add_argument("--ratio-threshold", type=float, default=1.5,
                        help="Min ratio of 2nd/1st Mahalanobis distance for clean assignment (default: 1.5)")
    parser.add_argument("--max-distance-percentile", type=float, default=95,
                        help="Within-cluster distance percentile for outlier detection (default: 95)")
    parser.add_argument("--split-signal-threshold", type=float, default=0.4,
                        help="Max fraction of PCs voting for a different subfamily (default: 0.4)")
    parser.add_argument("--min-copies", type=int, default=10,
                        help="Min copies nearest a reference in PCA space for presence "
                             "check (first filter before separation test) (default: 10)")
    parser.add_argument("--min-separation", type=float, default=1.0,
                        help="Min cluster separation ratio (centroid_dist / pooled_spread) "
                             "for a subfamily to be considered distinct and present; "
                             "subfamilies below this merge into the dominant one "
                             "(default: 1.0)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N random copies (for large datasets, default: use all)")
    parser.add_argument("--jobs", "-j", type=int, default=4,
                        help="Parallel MAFFT alignment jobs (default: 4)")
    parser.add_argument("--plot", action="store_true", help="Generate MSA-PCA scatter plot")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep ratio and split thresholds (requires --ground-truth)")
    args = parser.parse_args()

    # Validate argument combinations
    sinerella_mode = args.assignment is not None
    legacy_mode = args.scores is not None

    if sinerella_mode:
        if not args.extracted:
            parser.error("--extracted is required with --assignment")
        if not args.subfamilies:
            parser.error("--subfamilies is required with --assignment")
    elif legacy_mode:
        if not args.aligned:
            parser.error("--aligned is required with --scores (legacy mode)")
    else:
        parser.error("Provide either --assignment (SINEderella mode) or --scores (legacy mode)")

    print("=" * 70)
    print("SINE MSA-PCA DISCRIMINATOR")
    print("=" * 70)

    # Parse consensuses
    print("\nParsing consensus names...")
    consensus_seqs = parse_fasta(args.consensuses)
    if sinerella_mode:
        target_sfs = [s.strip() for s in args.subfamilies.split(',')]
        # Only keep consensuses that match requested subfamilies
        subfamilies = [sf for sf in target_sfs if sf in consensus_seqs]
        missing = set(target_sfs) - set(subfamilies)
        if missing:
            print(f"  WARNING: these subfamilies not in consensus file: {missing}")
        if not subfamilies:
            print("  ERROR: no target subfamilies found in consensus file")
            sys.exit(1)
    else:
        subfamilies = list(consensus_seqs.keys())
    print(f"  Subfamilies: {subfamilies}")

    if sinerella_mode:
        # === SINEderella input path ===
        print(f"\nParsing SINEderella assignment: {args.assignment}")
        bs_assignments, bs_scores = parse_assignment_tsv(args.assignment, subfamilies)
        print(f"  Copies assigned to target subfamilies: {len(bs_assignments)}")
        for sf in subfamilies:
            n = sum(1 for v in bs_assignments.values() if v == sf)
            print(f"    {sf}: {n} copies")

        if not bs_assignments:
            print("  ERROR: no copies found for target subfamilies")
            sys.exit(1)

        copy_names = list(bs_assignments.keys())

        # Sampling for large datasets
        if args.sample and len(copy_names) > args.sample:
            copy_names = random.sample(copy_names, args.sample)
            print(f"  Sampled: {len(copy_names)} copies (from {len(bs_assignments)} total)")

        # Warn about large datasets
        if len(copy_names) > 50000 and not args.aligned and not args.sample:
            print(f"\n  NOTE: {len(copy_names)} copies — pairwise alignment will take a while")
            print(f"  Each copy aligned individually (parallelizable with -j)")
            print(f"  Consider --sample N to test first (e.g. --sample 5000)")
            print()

        if args.aligned:
            print(f"\nUsing pre-aligned FASTA: {args.aligned}")
        else:
            # Extract target sequences and run MAFFT
            print(f"\nExtracting target copies from {args.extracted}...")
            target_seqs = extract_target_copies(args.extracted, copy_names)
            print(f"  Extracted: {len(target_seqs)} / {len(copy_names)} copies")
            # Update copy_names to only those actually found
            copy_names = [n for n in copy_names if n in target_seqs]
            if not copy_names:
                print("  ERROR: no target sequences found in extracted.fasta")
                sys.exit(1)

            # Write copies to temp file for MAFFT
            os.makedirs(args.outdir, exist_ok=True)
            copies_path = os.path.join(args.outdir, "_copies_for_mafft.fa")
            with open(copies_path, 'w', newline='\n') as f:
                for name in copy_names:
                    f.write(f">{name}\n{target_seqs[name]}\n")

            # Pairwise alignment: each copy aligned individually to consensus profile
            cons_aln_path = os.path.join(args.outdir, "_cons_aligned.fa")
            aligned_path = os.path.join(args.outdir, "_aligned.fa")

            print(f"\nPairwise-aligning {len(copy_names)} copies against consensus profile...")
            print(f"  Step 1: Pre-aligning {len(subfamilies)} consensuses...")
            ret = subprocess.run(
                f"mafft --auto {args.consensuses}",
                shell=True, capture_output=True
            )
            if ret.returncode != 0:
                print("  ERROR: mafft --auto failed on consensuses. Is MAFFT installed?")
                print(f"  stderr: {ret.stderr.decode()[:500]}")
                sys.exit(1)
            with open(cons_aln_path, 'wb') as f:
                f.write(ret.stdout)
            cons_aln_seqs = parse_fasta(cons_aln_path)
            print(f"  Consensus alignment done ({len(cons_aln_seqs)} seqs)")

            # Step 2: Pairwise align each copy to consensus profile
            print(f"  Step 2: Pairwise alignment ({args.jobs} parallel jobs)...")
            aligned_copies = pairwise_align_copies(
                cons_aln_path, copy_names, target_seqs,
                args.outdir, args.jobs
            )
            # Update copy_names to those that aligned successfully
            copy_names = [n for n in copy_names if n in aligned_copies]
            print(f"  Successfully aligned: {len(copy_names)} / {len(target_seqs)} copies")

            # Write combined alignment: consensuses + all aligned copies
            with open(aligned_path, 'w', newline='\n') as f:
                for name, seq in cons_aln_seqs.items():
                    f.write(f">{name}\n{seq}\n")
                for name in copy_names:
                    f.write(f">{name}\n{aligned_copies[name]}\n")
            print(f"  Combined alignment written ({len(cons_aln_seqs) + len(copy_names)} seqs)")

            args.aligned = aligned_path

    else:
        # === Legacy input path (ssearch36 scores) ===
        print("\nParsing bitscore assignments...")
        scores = parse_scores(args.scores)
        print(f"  Score pairs: {len(scores)}")

        # Identify copy names (everything in alignment that isn't a consensus)
        aln_seqs = parse_fasta(args.aligned)
        all_aln_names = list(aln_seqs.keys())
        copy_names = [n for n in all_aln_names if n not in subfamilies]
        print(f"  Copies in alignment: {len(copy_names)}")

        # Bitscore assignments from raw scores
        bs_assignments = get_bitscore_assignment(scores, subfamilies, copy_names)
        for sf in subfamilies:
            n = sum(1 for v in bs_assignments.values() if v == sf)
            print(f"    {sf}: {n} copies (bitscore)")

    # PCA
    print("\nRunning MSA-PCA...")
    coords, coords_2d, present, name_idx, ve, n_comp = compute_discrimination(
        args.aligned, subfamilies, copy_names, args.n_components
    )

    # Detect which subfamilies are genuinely present in this genome
    print("\nDetecting subfamily presence...")
    present_sfs, absent_sfs = detect_present_subfamilies(
        coords, name_idx, subfamilies, copy_names, bs_assignments,
        min_copies=args.min_copies, min_separation=args.min_separation
    )
    if absent_sfs:
        for sf, reason in absent_sfs.items():
            print(f"  ABSENT: '{sf}' — {reason}; excluded from discrimination")
        subfamilies = present_sfs
        if not subfamilies:
            print("  ERROR: no subfamilies detected as present in this genome")
            sys.exit(1)
        print(f"  Active subfamilies: {subfamilies}")
    else:
        print(f"  All {len(subfamilies)} subfamilies present "
              f"(each has \u2265 {args.min_copies} copies near its reference)")

    # Discriminate
    print("\nDiscriminating...")
    results = discriminate(
        coords, name_idx, subfamilies, copy_names, bs_assignments,
        ratio_threshold=args.ratio_threshold,
        max_distance_percentile=args.max_distance_percentile,
        split_signal_threshold=args.split_signal_threshold,
        variance_ratios=ve,
    )

    # Summary
    if not results:
        print("\n  ERROR: No copies to discriminate (all alignments may have failed).")
        sys.exit(1)
    clean = sum(1 for r in results if r['category'] == 'clean')
    grey_amb = sum(1 for r in results if r['category'] == 'grey_ambiguous')
    grey_out = sum(1 for r in results if r['category'] == 'grey_outlier')
    total = len(results)
    print(f"\n  Results:")
    print(f"    Clean:          {clean:4d} ({clean/total*100:.1f}%)")
    print(f"    Grey ambiguous: {grey_amb:4d} ({grey_amb/total*100:.1f}%)")
    print(f"    Grey outlier:   {grey_out:4d} ({grey_out/total*100:.1f}%)")
    print(f"    Total:          {total:4d}")

    # Distribution diagnostics
    ratios = [r['ratio'] for r in results if r['ratio'] < float('inf')]
    splits = [r.get('split_score', 0) for r in results]
    if ratios:
        r_arr = np.array(ratios)
        s_arr = np.array(splits)
        print(f"\n  Ratio distribution:  median={np.median(r_arr):.2f}, "
              f"Q25={np.percentile(r_arr, 25):.2f}, Q75={np.percentile(r_arr, 75):.2f}")
        print(f"  Split distribution:  median={np.median(s_arr):.2f}, "
              f"Q25={np.percentile(s_arr, 25):.2f}, Q75={np.percentile(s_arr, 75):.2f}")

    # Per-subfamily clean counts
    for sf in subfamilies:
        n = sum(1 for r in results if r['category'] == 'clean' and r['pca_sf'] == sf)
        print(f"    {sf} clean: {n}")

    # Write outputs
    print(f"\nWriting outputs to {args.outdir}/")
    write_outputs(results, args.outdir, args.aligned)

    # Ground truth validation
    if args.ground_truth:
        validate_vs_ground_truth(results, args.ground_truth)

    # Sweep mode — try many parameter combos against ground truth
    if args.sweep and args.ground_truth:
        print("\n" + "=" * 70)
        print("PARAMETER SWEEP")
        print("=" * 70)
        truth = {}
        with open(args.ground_truth, 'r') as f:
            f.readline()
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) >= 3:
                    truth[fields[0]] = {'sf': fields[1], 'category': fields[2]}

        print(f"{'ratio':>6} {'split':>6} | {'clean':>5} {'grey':>5} | "
              f"{'prec':>5} {'recall':>6} | {'caught':>6} {'escape':>6} | {'F1':>5}")
        print("-" * 75)

        best_f1 = 0
        best_params = None
        for ratio_t in [1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0]:
            for split_t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
                r = discriminate(coords, name_idx, subfamilies, copy_names,
                                 bs_assignments, ratio_threshold=ratio_t,
                                 max_distance_percentile=args.max_distance_percentile,
                                 split_signal_threshold=split_t,
                                 variance_ratios=ve)

                # Score: maximize (correct clean assignable) while penalizing (tricky escapes)
                assignable = [x for x in r if x['copy'] in truth
                              and truth[x['copy']]['category'] in ('core', 'divergent')]
                tricky = [x for x in r if x['copy'] in truth
                          and truth[x['copy']]['category'] in ('chimera', 'mimic')]

                correct_clean = sum(1 for x in assignable
                                    if x['category'] == 'clean'
                                    and x['pca_sf'] == truth[x['copy']]['sf'])
                wrong_clean = sum(1 for x in assignable
                                  if x['category'] == 'clean'
                                  and x['pca_sf'] != truth[x['copy']]['sf'])
                grey_assign = sum(1 for x in assignable if x['category'] != 'clean')
                caught = sum(1 for x in tricky if x['category'] != 'clean')
                escaped = len(tricky) - caught

                total_clean = sum(1 for x in r if x['category'] == 'clean')
                total_grey = len(r) - total_clean

                # Precision = correct_clean / (correct_clean + wrong_clean + escaped)
                # Recall = correct_clean / total_assignable
                denom_p = correct_clean + wrong_clean + escaped
                precision = correct_clean / denom_p if denom_p > 0 else 0
                recall = correct_clean / len(assignable) if assignable else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                print(f"{ratio_t:6.1f} {split_t:6.1f} | {total_clean:5d} {total_grey:5d} | "
                      f"{precision:5.1%} {recall:6.1%} | {caught:5d}/{len(tricky):<3d} "
                      f"{escaped:5d}/{len(tricky):<3d} | {f1:5.3f}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = (ratio_t, split_t, precision, recall, caught, escaped)

        if best_params:
            rt, st, p, rc, ca, es = best_params
            print(f"\n  BEST: ratio={rt}, split={st} -> "
                  f"precision={p:.1%}, recall={rc:.1%}, F1={best_f1:.3f}")
            print(f"         caught {ca} tricky, {es} escaped")

    # Optional plot
    if args.plot:
        try:
            import plotly.graph_objects as go
            print("\nGenerating discrimination plot...")
            fig = go.Figure()

            # Color map for categories
            cat_colors = {
                'clean': None,  # will use subfamily colors
                'grey_ambiguous': 'rgba(150,150,150,0.6)',
                'grey_outlier': 'rgba(80,80,80,0.4)',
            }
            _palette = ['#1f77b4', '#ff7f0e', '#2ca02c',
                        '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            sf_colors = {sf: _palette[i % len(_palette)]
                         for i, sf in enumerate(subfamilies)}

            # Plot clean copies by subfamily
            for sf in subfamilies:
                members = [r for r in results
                           if r['category'] == 'clean' and r['pca_sf'] == sf]
                if members:
                    x = [coords_2d[r['copy']][0] for r in members if r['copy'] in coords_2d]
                    y = [coords_2d[r['copy']][1] for r in members if r['copy'] in coords_2d]
                    names = [r['copy'] for r in members if r['copy'] in coords_2d]
                    fig.add_trace(go.Scatter(
                        x=x, y=y, mode='markers', name=f'{sf} (clean)',
                        marker=dict(color=sf_colors.get(sf, '#333'), size=8, opacity=0.8),
                        text=names, hovertemplate='%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}',
                    ))

            # Plot grey zone
            grey_results = [r for r in results if r['category'] != 'clean']
            if grey_results:
                x = [coords_2d[r['copy']][0] for r in grey_results if r['copy'] in coords_2d]
                y = [coords_2d[r['copy']][1] for r in grey_results if r['copy'] in coords_2d]
                names = [f"{r['copy']} ({r['category']}, ratio={r['ratio']:.2f})"
                         for r in grey_results if r['copy'] in coords_2d]
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode='markers', name='Grey zone',
                    marker=dict(color='grey', size=10, opacity=0.5,
                                symbol='diamond', line=dict(color='black', width=1)),
                    text=names, hovertemplate='%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}',
                ))

            # Plot consensus positions
            for sf in subfamilies:
                if sf in coords_2d:
                    fig.add_trace(go.Scatter(
                        x=[coords_2d[sf][0]], y=[coords_2d[sf][1]],
                        mode='markers+text', name=f'{sf} consensus',
                        marker=dict(color=sf_colors.get(sf, '#333'), size=16,
                                    symbol='star', line=dict(color='black', width=2)),
                        text=[sf], textposition='top center',
                        hovertemplate=f'{sf} consensus<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}',
                    ))

            fig.update_layout(
                title=f"MSA-PCA Discrimination (PC1={ve[0]:.1%}, PC2={ve[1]:.1%})",
                xaxis_title=f"PC1 ({ve[0]:.1%} variance)",
                yaxis_title=f"PC2 ({ve[1]:.1%} variance)",
                template='plotly_white',
                width=1000, height=800,
            )
            plot_path = os.path.join(args.outdir, "discrimination_plot.html")
            fig.write_html(plot_path)
            print(f"  Plot: {plot_path}")
        except ImportError:
            print("  plotly not installed, skipping plot")

    print("\nDone!")


if __name__ == "__main__":
    main()
