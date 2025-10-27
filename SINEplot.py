#!/usr/bin/env python3
"""
SINE Bitscore Visualizer - FINAL FIXED VERSION
CRITICAL FIXES:
- Parser: keeps MAX bitscore per (query, subject) pair
- Assignment: uses ABSOLUTE BITSCORES (not normalized %)
- Placement: uses ABSOLUTE BITSCORES for weighted centroid (Option A)

Preserves all enhancements:
- Subfamily vs Ternary mode
- Label overlap avoidance
- Smart downsampling & clustering
- Dot size slider, clipboard selection, etc.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import MDS
from sklearn.cluster import DBSCAN
import argparse
import time
import os
from itertools import combinations

def parse_ssearch_output(filename):
    """Parse ssearch36 output and keep MAX bitscore per (query, subject) pair."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Input file not found: {filename}")
    print(f"Parsing {filename}...")
    score_dict = {}
    line_count = 0
    with open(filename, 'r') as f:
        for line in f:
            line_count += 1
            if line_count % 10000 == 0:
                print(f"  Processed {line_count} lines...")
            line = line.strip()
            if not line or any(x in line for x in ['working', 'Cycle', '[INFO]', 
                                                    'less than', 'searching', 
                                                    'cat', '(base)']):
                continue
            fields = line.split()
            if len(fields) >= 12:
                try:
                    query = fields[0]
                    subject = fields[1]
                    bitscore = float(fields[11])
                    key = (query, subject)
                    if key not in score_dict or bitscore > score_dict[key]:
                        score_dict[key] = bitscore
                except (ValueError, IndexError):
                    continue
    data = [[q, s, b] for (q, s), b in score_dict.items()]
    df = pd.DataFrame(data, columns=['query', 'subject', 'bitscore'])
    print(f"  Total unique (query, subject) pairs: {len(df)}")
    return df

def get_subfamily_distances(df, subfamilies):
    """Calculate distance matrix between subfamilies."""
    n = len(subfamilies)
    distances = np.zeros((n, n))
    max_scores = {}
    for sf in subfamilies:
        self_align = df[(df['query'] == sf) & (df['subject'] == sf)]
        if len(self_align) > 0:
            max_scores[sf] = self_align['bitscore'].max()
        else:
            best_score = df[df['query'] == sf]['bitscore'].max()
            max_scores[sf] = best_score * 1.2 if best_score > 0 else 300

    for i, sf1 in enumerate(subfamilies):
        for j, sf2 in enumerate(subfamilies):
            if i == j:
                distances[i, j] = 0
            elif i < j:
                cross_align1 = df[(df['query'] == sf1) & (df['subject'] == sf2)]
                cross_align2 = df[(df['query'] == sf2) & (df['subject'] == sf1)]
                scores = []
                if len(cross_align1) > 0:
                    scores.append(cross_align1['bitscore'].max())
                if len(cross_align2) > 0:
                    scores.append(cross_align2['bitscore'].max())
                score = max(scores) if scores else 0
                avg_max = (max_scores[sf1] + max_scores[sf2]) / 2
                dist = avg_max - score
                distances[i, j] = dist
                distances[j, i] = dist
    return distances, max_scores

def get_subfamily_positions_phylo(df, subfamilies):
    """Position subfamilies using MDS."""
    distances, max_scores = get_subfamily_distances(df, subfamilies)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distances)
    coords = coords / np.max(np.abs(coords)) * 10
    positions = {sf: (coords[i, 0], coords[i, 1]) for i, sf in enumerate(subfamilies)}
    return positions, max_scores

def get_subfamily_positions_geometric(subfamilies):
    """Position subfamilies in geometric pattern."""
    n = len(subfamilies)
    positions = {}
    if n == 2:
        positions[subfamilies[0]] = (-10, 0)
        positions[subfamilies[1]] = (10, 0)
    else:
        radius = 10
        for i, sf in enumerate(subfamilies):
            angle = 2 * np.pi * i / n
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions[sf] = (x, y)
    return positions

def position_sine_absolute_weighted(scores, sf_positions):
    """
    FIXED: Use ratio-based placement to properly handle dominant subfamilies
    """
    if sum(scores.values()) == 0:
        centroid = np.mean(list(sf_positions.values()), axis=0)
        return tuple(centroid)
    
    # Get top 2 scores
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) < 2:
        best_sf, best_score = sorted_scores[0]
        return sf_positions[best_sf]
    
    best_sf, best_score = sorted_scores[0]
    second_sf, second_score = sorted_scores[1]
    
    # Calculate dominance ratio
    ratio = best_score / second_score if second_score > 0 else float('inf')
    
    # Get positions
    best_x, best_y = sf_positions[best_sf]
    
    # Calculate weighted centroid
    total = sum(scores.values())
    centroid_x = sum(scores[sf] * sf_positions[sf][0] for sf in scores) / total
    centroid_y = sum(scores[sf] * sf_positions[sf][1] for sf in scores) / total
    
    # Blend based on ratio
    if ratio > 10:  
        blend = 0.95
    elif ratio > 5:  
        blend = 0.85  
    elif ratio > 2:  
        blend = 0.70
    else:  
        blend = 0.50
    
    x = blend * best_x + (1 - blend) * centroid_x
    y = blend * best_y + (1 - blend) * centroid_y
    
    return (x, y)

def avoid_label_overlap(positions, subfamilies, min_distance=2.5):
    label_positions = {}
    centroid = np.mean(list(positions.values()), axis=0)
    for sf in subfamilies:
        x, y = positions[sf]
        direction = np.array([x, y]) - centroid
        if np.linalg.norm(direction) == 0:
            direction = np.array([1, 0])
        direction = direction / np.linalg.norm(direction)
        label_positions[sf] = np.array([x, y]) + direction * 1.3

    for _ in range(20):
        moved = False
        for a, b in combinations(subfamilies, 2):
            pos_a, pos_b = label_positions[a], label_positions[b]
            diff = pos_a - pos_b
            dist = np.linalg.norm(diff)
            if 0 < dist < min_distance:
                adjust = (min_distance - dist) / 2
                unit = diff / dist
                label_positions[a] += unit * adjust
                label_positions[b] -= unit * adjust
                moved = True
        if not moved:
            break
    return label_positions

def smart_downsample(sine_data, max_points=1000):
    if len(sine_data) <= max_points:
        return sine_data, None
    print(f"\nDataset has {len(sine_data)} sequences, downsampling to ~{max_points} for display...")
    subfamilies = list(set(s['sf'] for s in sine_data))
    selected = []
    points_per_sf = max_points // len(subfamilies)
    for sf in subfamilies:
        sf_sines = [s for s in sine_data if s['sf'] == sf]
        if len(sf_sines) <= points_per_sf:
            selected.extend(sf_sines)
            continue
        sf_sines.sort(key=lambda x: x['intensity'], reverse=True)
        n_top = max(1, int(len(sf_sines) * 0.2))
        selected.extend(sf_sines[:n_top])
        n_bottom = max(1, int(len(sf_sines) * 0.1))
        selected.extend(sf_sines[-n_bottom:])
        remaining = sf_sines[n_top:-n_bottom]
        n_random = min(len(remaining), points_per_sf - n_top - n_bottom)
        if n_random > 0:
            indices = np.random.choice(len(remaining), n_random, replace=False)
            selected.extend([remaining[i] for i in indices])
    print(f"  Selected {len(selected)} representative sequences")
    print(f"  All {len(sine_data)} sequences remain in the dataset for statistics")
    full_data = sine_data
    if len(selected) < len(full_data):
        return selected, full_data
    else:
        return selected, None

def cluster_dense_regions(sine_data, eps=0.5, min_samples=5):
    if len(sine_data) < 5000:
        return sine_data, None
    print(f"\nClustering {len(sine_data)} points for density analysis...")
    X = np.array([[s['x'], s['y']] for s in sine_data])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    print(f"  Found {n_clusters} dense clusters")
    clustered_data = []
    for label in set(clustering.labels_):
        if label == -1:
            cluster_points = [sine_data[i] for i in range(len(sine_data)) if clustering.labels_[i] == label]
            clustered_data.extend(cluster_points)
        else:
            cluster_points = [sine_data[i] for i in range(len(sine_data)) if clustering.labels_[i] == label]
            cluster_points.sort(key=lambda x: x['intensity'], reverse=True)
            clustered_data.extend(cluster_points[:min(5, len(cluster_points))])
    print(f"  Reduced to {len(clustered_data)} representative points")
    full_data = sine_data
    if len(clustered_data) < len(full_data):
        return clustered_data, full_data
    else:
        return clustered_data, None

def add_ternary_color_guide(fig):
    fig.add_annotation(
        text='<b>Color Mixing Guide:</b>',
        xref='paper', yref='paper',
        x=1.15, y=0.5,
        showarrow=False,
        font=dict(size=12, color='black'),
        xanchor='left'
    )
    color_samples = [
        ('Pure dD', 'rgb(255,0,0)', 0.45),
        ('Pure dE', 'rgb(0,0,255)', 0.40),
        ('Pure dF', 'rgb(0,255,0)', 0.35),
        ('dD+dE mix', 'rgb(128,0,128)', 0.28),
        ('dD+dF mix', 'rgb(128,128,0)', 0.23),
        ('dE+dF mix', 'rgb(0,128,128)', 0.18),
        ('All 3 equal', 'rgb(85,85,85)', 0.13),
    ]
    for label, color, y_pos in color_samples:
        fig.add_annotation(
            text=f'в—Џ {label}',
            xref='paper', yref='paper',
            x=1.15, y=y_pos,
            showarrow=False,
            font=dict(size=10, color=color),
            xanchor='left'
        )
    return fig

def create_interactive_plot(df, mode='phylo', output='sine_interactive.html', 
                           title='SINE Bitscore Distribution', color_mode='subfamily',
                           max_display_points=1000, use_clustering=False):
    start_time = time.time()
    all_queries = df['query'].unique()
    all_subjects = df['subject'].unique()
    query_counts = df['query'].value_counts()
    subfamilies = [q for q in all_queries if len(q) < 15 and query_counts[q] > 5]
    if len(subfamilies) == 0:
        subfamilies = [q for q in all_queries if any((df['query'] == q) & (df['subject'] == q))]
    sines = [s for s in all_subjects if s not in subfamilies]
    print(f"Found {len(subfamilies)} subfamilies: {subfamilies}")
    print(f"Found {len(sines)} SINE loci")

    if len(sines) > 5000:
        print(f"\nвљЎ Large dataset detected ({len(sines)} sequences)")
        print(f"   Early sampling {max_display_points*2} for processing...")
        import random
        random.seed(42)
        sines_to_process = random.sample(sines, min(max_display_points*2, len(sines)))
        all_sines = sines
    else:
        sines_to_process = sines
        all_sines = sines

    max_scores = {}
    for sf in subfamilies:
        self_align = df[(df['query'] == sf) & (df['subject'] == sf)]
        if len(self_align) > 0:
            max_scores[sf] = self_align['bitscore'].max()
            print(f"  {sf}: max score = {max_scores[sf]:.1f}")
        else:
            max_scores[sf] = df[df['query'] == sf]['bitscore'].max() * 1.2
            print(f"  {sf}: estimated max score = {max_scores[sf]:.1f}")

    if mode == 'phylo':
        sf_positions, max_scores = get_subfamily_positions_phylo(df, subfamilies)
    else:
        sf_positions = get_subfamily_positions_geometric(subfamilies)

    base_colors = ['#FF4444', '#4444FF', '#44FF44', '#FF44FF', '#FFAA44']

    print("\nBuilding bitscore lookup...")
    bitscore_lookup = {}
    for sf in subfamilies:
        sf_data = df[df['query'] == sf]
        bitscore_lookup[sf] = dict(zip(sf_data['subject'], sf_data['bitscore']))

    print(f"Positioning {len(sines_to_process)} sequences...")
    sine_data = []
    for i, sine in enumerate(sines_to_process):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(sines_to_process)} sequences...", end='\r')
        scores = {sf: bitscore_lookup[sf].get(sine, 0) for sf in subfamilies}
        pos = position_sine_absolute_weighted(scores, sf_positions)  # This will now work better!

        # вњ… FIXED: Assignment uses ABSOLUTE BITSCORES
        if sum(scores.values()) > 0:
            dominant_sf = max(scores, key=scores.get)
            sf_idx = subfamilies.index(dominant_sf)
            # Intensity = normalized score (for confidence/opacity)
            intensity = scores[dominant_sf] / max_scores[dominant_sf] if max_scores[dominant_sf] > 0 else 0
            intensity = max(0.0, min(1.0, intensity))
        else:
            dominant_sf = 'unknown'
            sf_idx = -1
            intensity = 0

        sine_data.append({
            'id': sine,
            'x': pos[0],
            'y': pos[1],
            'sf': dominant_sf,
            'sf_idx': sf_idx,
            'intensity': intensity,
            'scores': scores
        })
    print(f"\n  Positioned {len(sine_data)} sequences")

    full_sine_data = sine_data
    if use_clustering:
        sine_data, full_data = cluster_dense_regions(sine_data)
        if full_data is not None:
            full_sine_data = full_data
    elif len(sine_data) > max_display_points:
        sine_data, full_data = smart_downsample(sine_data, max_display_points)
        if full_data is not None:
            full_sine_data = full_data

    all_x_vals = [s['x'] for s in sine_data]
    all_y_vals = [s['y'] for s in sine_data]
    all_x_vals.extend([x for x, y in sf_positions.values()])
    all_y_vals.extend([y for x, y in sf_positions.values()])
    x_min, x_max = min(all_x_vals), max(all_x_vals)
    y_min, y_max = min(all_y_vals), max(all_y_vals)
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = max(x_range, y_range) * 0.15
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    fig = go.Figure()
    subfamily_trace_indices = []
    for sf_idx, sf in enumerate(subfamilies):
        sf_sines = [s for s in sine_data if s['sf'] == sf]
        if not sf_sines:
            print(f"  WARNING: No SINEs assigned to {sf}")
            continue
        x_vals = [s['x'] for s in sf_sines]
        y_vals = [s['y'] for s in sf_sines]
        hover_texts = []
        for s in sf_sines:
            text_lines = [f"<b>{s['id']}</b><br>"]
            text_lines.append(f"<b>Assigned to: {s['sf']}</b><br>")
            text_lines.append(f"<b>Confidence: {s['intensity']*100:.1f}%</b><br>")
            text_lines.append("<br><b>Bitscores (raw / %):</b><br>")
            for sub_sf in subfamilies:
                score = s['scores'][sub_sf]
                pct = (score / max_scores[sub_sf]) * 100 if max_scores[sub_sf] > 0 else 0
                marker = "в—Џ " if sub_sf == s['sf'] else "  "
                text_lines.append(f"{marker}{sub_sf}: <b>{score:.1f}</b> bits ({pct:.1f}%)<br>")
            hover_texts.append(''.join(text_lines))
        colors = []
        for s in sf_sines:
            base = base_colors[sf_idx % len(base_colors)]
            intensity = s['intensity']
            rgb = tuple(int(base[i:i+2], 16) for i in (1, 3, 5))
            rgba = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {0.3 + intensity*0.6})'
            colors.append(rgba)
        trace_type = go.Scattergl if len(sine_data) > 300 else go.Scatter
        fig.add_trace(trace_type(
            x=x_vals,
            y=y_vals,
            mode='markers',
            name=sf,
            marker=dict(
                size=10 if len(sine_data) > 300 else 12,
                color=colors,
                line=dict(width=0.5, color='rgba(0,0,0,0.3)')
            ),
            hovertext=hover_texts,
            hoverinfo='text',
            text=[s['id'] for s in sf_sines],
            visible=True
        ))
        subfamily_trace_indices.append(len(fig.data) - 1)

    all_x = [s['x'] for s in sine_data]
    all_y = [s['y'] for s in sine_data]
    ternary_colors = []
    hover_texts_ternary = []
    labels = []
    available_subfamilies = subfamilies[:3]
    sf_to_rgb = {}
    if len(available_subfamilies) >= 1:
        sf_to_rgb[available_subfamilies[0]] = (255, 0, 0)
    if len(available_subfamilies) >= 2:
        sf_to_rgb[available_subfamilies[1]] = (0, 0, 255)
    if len(available_subfamilies) >= 3:
        sf_to_rgb[available_subfamilies[2]] = (0, 255, 0)

    for s in sine_data:
        scores = s['scores']
        norm_scores = {sf: scores[sf]/max_scores[sf] if max_scores[sf] > 0 else 0 for sf in available_subfamilies}
        total = sum(norm_scores.values())
        if total > 0:
            weights = {sf: norm_scores[sf]/total for sf in available_subfamilies}
        else:
            weights = {sf: 1/len(available_subfamilies) for sf in available_subfamilies}
        r = sum(weights.get(sf, 0) * sf_to_rgb.get(sf, (0,0,0))[0] for sf in available_subfamilies)
        g = sum(weights.get(sf, 0) * sf_to_rgb.get(sf, (0,0,0))[1] for sf in available_subfamilies)
        b = sum(weights.get(sf, 0) * sf_to_rgb.get(sf, (0,0,0))[2] for sf in available_subfamilies)
        rgba = f'rgba({int(r)}, {int(g)}, {int(b)}, 0.75)'
        ternary_colors.append(rgba)
        text_lines = [f"<b>{s['id']}</b><br>"]
        text_lines.append(f"<b>Best match: {s['sf']}</b><br>")
        text_lines.append("<br><b>Relative contribution:</b><br>")
        for sf in available_subfamilies:
            rgb_val = sf_to_rgb.get(sf, (128,128,128))
            rgb_str = f"rgb({rgb_val[0]},{rgb_val[1]},{rgb_val[2]})"
            text_lines.append(f"<span style='color:{rgb_str}'>в—Џ</span> {sf}: {weights[sf]*100:.1f}%<br>")
        text_lines.append("<br><b>Raw bitscores:</b><br>")
        for sf in subfamilies:
            score = scores[sf]
            pct = (score / max_scores[sf]) * 100 if max_scores[sf] > 0 else 0
            text_lines.append(f"{sf}: <b>{score:.1f}</b> bits ({pct:.1f}%)<br>")
        hover_texts_ternary.append(''.join(text_lines))
        labels.append(s['id'])

    trace_type = go.Scattergl if len(sine_data) > 300 else go.Scatter
    fig.add_trace(trace_type(
        x=all_x,
        y=all_y,
        mode='markers',
        name='Ternary Blend',
        marker=dict(
            size=10 if len(sine_data) > 300 else 12,
            color=ternary_colors,
            line=dict(width=0.5 if len(sine_data) > 300 else 0.8, color='rgba(0,0,0,0.4)')
        ),
        hovertext=hover_texts_ternary,
        hoverinfo='text',
        text=labels,
        visible=False,
        showlegend=False
    ))
    ternary_trace_index = len(fig.data) - 1

    label_positions = avoid_label_overlap(sf_positions, subfamilies)
    for i, sf in enumerate(subfamilies):
        x, y = sf_positions[sf]
        color = base_colors[i % len(base_colors)]
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(
                size=16,
                color=color,
                symbol='diamond',
                line=dict(width=2, color='black')
            ),
            showlegend=False,
            hoverinfo='text',
            hovertext=f'<b>Subfamily: {sf}</b><br>Consensus position'
        ))
        label_x, label_y = label_positions[sf]
        fig.add_trace(go.Scatter(
            x=[label_x], y=[label_y],
            mode='text',
            text=[sf],
            textposition='middle center',
            textfont=dict(size=16, color='black', family='Arial Black'),
            showlegend=False,
            hoverinfo='skip'
        ))

    total_traces = len(fig.data)
    subtitle = ('<sub>Hover for details<br>' +
               'Click legend to toggle<br>' +
               '"Reset Legend" restores all</sub><br>' +
               '<sub><b>Subfamily Mode:</b><br>Color = assignment<br>Opacity = confidence</sub><br>' +
               '<sub><b>Ternary Mode:</b><br>RGB blend of top 3</sub>')
    if len(sine_data) < len(full_sine_data):
        subtitle += f'<br><sub>Showing {len(sine_data)} of {len(full_sine_data)}</sub>'

    fig.update_layout(
        xaxis=dict(
            title='Dimension 1',
            range=[x_min, x_max],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.3)'
        ),
        yaxis=dict(
            title='Dimension 2',
            range=[y_min, y_max],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            scaleanchor='x',
            scaleratio=1,
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.3)'
        ),
        hovermode='closest',
        dragmode='zoom',
        width=1500,
        height=1000,
        plot_bgcolor='white',
        margin=dict(l=80, r=350, t=60, b=80),
        legend=dict(
            title='Subfamilies',
            yanchor='top',
            y=0.65,
            xanchor='left',
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1,
            tracegroupgap=5
        )
    )

    fig.add_annotation(
        text=f'<b>{title}</b>',
        xref='paper', yref='paper',
        x=1.02, y=0.98,
        xanchor='left', yanchor='top',
        showarrow=False,
        font=dict(size=16, color='black')
    )
    fig.add_annotation(
        text=subtitle,
        xref='paper', yref='paper',
        x=1.02, y=0.93,
        xanchor='left', yanchor='top',
        showarrow=False,
        font=dict(size=9, color='#444'),
        align='left'
    )

    if subfamily_trace_indices:
        visible_args_show_all = {'visible': [True if i in subfamily_trace_indices or i >= total_traces - len(subfamilies)*2 else False for i in range(total_traces)]}
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="down",
                    buttons=[
                        dict(
                            args=[
                                {"visible": [True if i in subfamily_trace_indices or i >= total_traces - len(subfamilies)*2 else False for i in range(total_traces)]},
                                {"xaxis.range": [x_min, x_max], "yaxis.range": [y_min, y_max]}
                            ],
                            label="Subfamily Mode",
                            method="update"
                        ),
                        dict(
                            args=[
                                {"visible": [i == ternary_trace_index or i >= total_traces - len(subfamilies)*2 for i in range(total_traces)]},
                                {"xaxis.range": [x_min, x_max], "yaxis.range": [y_min, y_max]}
                            ],
                            label="Ternary Mode",
                            method="update"
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=1.02,
                    xanchor="left",
                    y=0.80,
                    yanchor="top"
                ),
                dict(
                    type="buttons",
                    direction="down",
                    buttons=[
                        dict(
                            args=[visible_args_show_all],
                            label='Reset Legend',
                            method='update'
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=1.02,
                    xanchor="left",
                    y=0.45,
                    yanchor="top"
                )
            ]
        )

    if len(sine_data) > 2000:
        sizes = [(1.5, 'Mini'), (2, 'Micro'), (3, 'Tiny'), (4, 'X-Small'), (6, 'Small'), (8, 'Medium'), (11, 'Large')]
        default = 3
    elif len(sine_data) > 1000:
        sizes = [(2, 'Mini'), (3, 'Micro'), (4, 'Tiny'), (6, 'X-Small'), (8, 'Small'), (10, 'Medium'), (13, 'Large')]
        default = 3
    elif len(sine_data) > 500:
        sizes = [(3, 'Mini'), (4, 'Micro'), (6, 'Tiny'), (8, 'X-Small'), (10, 'Small'), (13, 'Medium'), (16, 'Large')]
        default = 3
    else:
        sizes = [(4, 'Mini'), (5, 'Micro'), (7, 'Tiny'), (10, 'X-Small'), (13, 'Small'), (16, 'Medium'), (20, 'Large')]
        default = 3

    num_subfamilies = len(subfamilies)
    data_trace_indices = set(subfamily_trace_indices + [ternary_trace_index])
    def create_size_list(base_size):
        size_list = []
        for i in range(len(fig.data)):
            if i in data_trace_indices:
                size_list.append(base_size)
            elif i >= len(fig.data) - num_subfamilies * 2:
                marker_idx = (i - (len(fig.data) - num_subfamilies * 2)) % 2
                if marker_idx == 0:
                    size_list.append(16)
                else:
                    size_list.append(None)
            else:
                size_list.append(base_size)
        return size_list

    steps = [{'label': label, 'method': 'restyle', 
              'args': [{'marker.size': create_size_list(size)}]} 
             for size, label in sizes]
    fig.update_layout(
        sliders=[{
            'active': default,
            'currentvalue': {"prefix": "Dot Size: ", "font": {"size": 11}, "visible": True, "xanchor": "left"},
            'pad': {"b": 10, "t": 10, "l": 0},
            'len': 0.15,
            'x': 1.02,
            'xanchor': 'left',
            'y': 0.20,
            'yanchor': 'top',
            'steps': steps,
            'bgcolor': 'rgba(240,240,240,0.9)',
            'bordercolor': '#aaa',
            'borderwidth': 1,
            'font': {'size': 9}
        }]
    )

    html_content = fig.to_html(include_plotlyjs='cdn')
    javascript_code = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        let plotDiv = document.querySelector('.js-plotly-plot');
        if (!plotDiv) return;
        function enhanceLegend() {
            let legendTexts = plotDiv.querySelectorAll('.legendtext');
            legendTexts.forEach((textElem, idx) => {
                let traceIdx = idx;
                if (traceIdx >= plotDiv.data.length || !plotDiv.data[traceIdx] || !plotDiv.data[traceIdx].x || plotDiv.data[traceIdx].x.length === 0) return;
                let isHidden = plotDiv.data[traceIdx].visible === 'legendonly';
                let currentText = textElem.textContent;
                if (currentText.endsWith(' [hidden]')) {
                    textElem.textContent = currentText.replace(' [hidden]', '');
                }
                if (isHidden) {
                    textElem.textContent += ' [hidden]';
                    textElem.setAttribute('fill', 'gray');
                } else {
                    textElem.setAttribute('fill', 'black');
                }
            });
        }
        setTimeout(enhanceLegend, 100);
        plotDiv.on('plotly_legendclick', function(data) {
            let idx = data.curveNumber;
            if (idx === undefined || !plotDiv.data[idx].x || plotDiv.data[idx].x.length === 0) return true;
            let currentVis = plotDiv.data[idx].visible;
            let newVis = (currentVis === true) ? 'legendonly' : true;
            Plotly.restyle(plotDiv, {visible: newVis}, [idx]);
            setTimeout(enhanceLegend, 50);
            return false;
        });
        plotDiv.on('plotly_restyle', function() {
            setTimeout(enhanceLegend, 50);
        });
        let deselectBtn = null;
        plotDiv.on('plotly_selected', function(eventData) {
            if (!eventData || !eventData.points || eventData.points.length === 0) {
                if (deselectBtn) { deselectBtn.remove(); deselectBtn = null; }
                return;
            }
            let ids = eventData.points.map(p => p.text).filter(t => t);
            if (ids.length === 0) return;
            navigator.clipboard.writeText(ids.join('\\n')).then(() => {
                let notif = document.createElement('div');
                notif.textContent = `OK Copied ${ids.length} sequence names!`;
                notif.style.cssText = 'position:fixed;top:20px;right:20px;padding:15px 25px;background:#2ecc71;color:white;border-radius:5px;font-weight:bold;z-index:10000;box-shadow:0 4px 6px rgba(0,0,0,0.3)';
                document.body.appendChild(notif);
                setTimeout(() => document.body.removeChild(notif), 3000);
            }).catch(err => console.error('Copy failed:', err));
            if (!deselectBtn) {
                deselectBtn = document.createElement('button');
                deselectBtn.textContent = 'Deselect';
                deselectBtn.style.cssText = 'position:fixed;top:70px;right:20px;padding:10px 20px;background:#e74c3c;color:white;border:none;border-radius:5px;font-weight:bold;cursor:pointer;z-index:10000;box-shadow:0 4px 6px rgba(0,0,0,0.3)';
                deselectBtn.onmouseover = () => { this.style.background = '#c0392b'; };
                deselectBtn.onmouseout = () => { this.style.background = '#e74c3c'; };
                deselectBtn.onclick = () => {
                    Plotly.restyle(plotDiv, {selectedpoints: [null]});
                    if (deselectBtn) { deselectBtn.remove(); deselectBtn = null; }
                };
                document.body.appendChild(deselectBtn);
            }
        });
        plotDiv.on('plotly_deselect', function() {
            if (deselectBtn) { deselectBtn.remove(); deselectBtn = null; }
        });
    });
    </script>
    """
    html_content = html_content.replace('</body>', f'{javascript_code}</body>')
    with open(output, 'w', encoding='utf-8') as f:
        f.write(html_content)

    elapsed_time = time.time() - start_time
    print(f"\nInteractive plot saved to {output}")
    print(f"  Processing time: {elapsed_time:.2f} seconds")
    print(f"  Displaying: {len(sine_data)} sequences")
    print(f"  Total dataset: {len(full_sine_data)} sequences")
    print("\nInteractive features:")
    print("  - Hover over dots to see bitscores")
    print("  - Click legend items to toggle group visibility")
    print("  - Click 'Reset Legend' to restore all hidden groups")
    print("  - Box/Lasso select copies sequence names to clipboard")
    print("  - Scroll/pinch to zoom, drag to pan, double-click to reset")
    print("  - Use mode buttons to switch visualization")
    print("  - Use 'Dot Size' slider to adjust marker sizes")
    print("\nTip: Zoom into crowded regions to see overlapping dots individually!")
    return fig, full_sine_data, sf_positions, max_scores

def print_stats(sine_data, subfamilies, max_scores):
    print("\n" + "="*60)
    print("COMPREHENSIVE STATISTICS")
    print("="*60)
    total_sines = len(sine_data)
    for sf in subfamilies:
        sf_sines = [s for s in sine_data if s['sf'] == sf]
        print(f"\n{sf}: {len(sf_sines)} SINEs ({len(sf_sines)/total_sines*100:.1f}%)")
        if len(sf_sines) > 0:
            similarities = [(s['scores'][sf] / max_scores[sf]) * 100 for s in sf_sines]
            avg_pct = np.mean(similarities)
            print(f"  Average similarity: {avg_pct:.1f}%")
            print(f"  Range: {min(similarities):.1f}% - {max(similarities):.1f}%")
            print(f"  Median: {np.median(similarities):.1f}%")
            print(f"  Std dev: {np.std(similarities):.1f}%")
            high_conf = sum(1 for s in similarities if s >= 60)
            med_conf = sum(1 for s in similarities if 30 <= s < 60)
            low_conf = sum(1 for s in similarities if s < 30)
            print(f"  Confidence distribution:")
            print(f"    High (>=60%): {high_conf} ({high_conf/len(sf_sines)*100:.0f}%)")
            print(f"    Medium (30-60%): {med_conf} ({med_conf/len(sf_sines)*100:.0f}%)")
            print(f"    Low (<30%): {low_conf} ({low_conf/len(sf_sines)*100:.0f}%)")
            x_vals = [s['x'] for s in sf_sines]
            y_vals = [s['y'] for s in sf_sines]
            print(f"  Position spread:")
            print(f"    X: [{min(x_vals):.1f}, {max(x_vals):.1f}]")
            print(f"    Y: [{min(y_vals):.1f}, {max(y_vals):.1f}]")
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Create interactive SINE bitscore visualization (FINAL FIXED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sine_visualizer_fixed_final.py score.txt -o plot.html
  python sine_visualizer_fixed_final.py score.txt --max-points 500
  python sine_visualizer_fixed_final.py score.txt --color-mode ternary
        """)
    parser.add_argument('input', help='ssearch36 output file')
    parser.add_argument('--mode', choices=['phylo', 'geometric'], default='phylo')
    parser.add_argument('-o', '--output', default='sine_interactive.html')
    parser.add_argument('-t', '--title', default='SINE Bitscore Distribution')
    parser.add_argument('--color-mode', choices=['subfamily', 'ternary'], default='subfamily')
    parser.add_argument('--max-points', type=int, default=1000)
    parser.add_argument('--cluster', action='store_true')
    args = parser.parse_args()

    print("="*60)
    print("SINE VISUALIZER - FINAL FIXED VERSION")
    print("="*60)
    print(f"\nReading data from {args.input}...")
    try:
        df = parse_ssearch_output(args.input)
    except Exception as e:
        print(f"Error: {e}")
        return

    n_subjects = len(df['subject'].unique())
    if args.max_points == 1000:
        if n_subjects > 20000:
            max_display = 500
        elif n_subjects > 10000:
            max_display = 750
        else:
            max_display = args.max_points
    else:
        max_display = args.max_points if args.max_points > 0 else 999999

    fig, sine_data, sf_pos, max_scores = create_interactive_plot(
        df,
        mode=args.mode,
        output=args.output,
        title=args.title,
        color_mode=args.color_mode,
        max_display_points=max_display,
        use_clustering=args.cluster
    )

    all_queries = df['query'].unique()
    query_counts = df['query'].value_counts()
    subfamilies = [q for q in all_queries if len(q) < 15 and query_counts[q] > 5]
    print_stats(sine_data, subfamilies, max_scores)
    print("\n Done!")

if __name__ == '__main__':
    main()
