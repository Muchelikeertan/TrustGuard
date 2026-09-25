"""Deterministic Graph Visualizer for TrustGuard Multimodal Contradiction Map.

Renders responsive, non-overlapping SVG relationship graphs showing:
1. Claimed Identity as the central reference node.
2. ONLY modalities that were actually supplied and analyzed (dynamic evidence filtering).
3. Labeled relationship edges (Agrees, Contradicts, Borderline, Inconclusive) between
   evidence nodes and the identity, plus cross-modal correlation edges.
4. Pure SVG rendering for 100% deterministic geometry across screen resolutions.
"""

from typing import Any, Dict, List, Optional, Tuple


def _is_modality_active(key: str, status: str, details: Dict[str, Any]) -> bool:
    """Determines whether a modality was actually supplied and evaluated."""
    if key == "image":
        img_det = details.get("image", {})
        if img_det.get("distance") is not None:
            return True
        reason = img_det.get("source_reason") or img_det.get("reason", "")
        if reason and "no image evidence" not in reason.lower() and "no photo" not in reason.lower():
            return True
        return status in ("PASS", "WARNING", "FAIL")

    elif key == "voice":
        voc_det = details.get("voice", {})
        if voc_det.get("similarity") is not None:
            return True
        reason = voc_det.get("source_reason") or voc_det.get("reason", "")
        if reason and "no audio evidence" not in reason.lower() and "no voice" not in reason.lower():
            return True
        return status in ("PASS", "WARNING", "FAIL")

    elif key == "chat":
        chat_det = details.get("chat_signal", {})
        return chat_det.get("final_score") is not None or status in ("PASS", "WARNING", "FAIL")

    elif key == "transcript":
        tr_det = details.get("transcript_signal", {})
        return tr_det.get("final_score") is not None or status in ("PASS", "WARNING", "FAIL")

    return False


def render_contradiction_graph_svg(
    claimed_identity: str,
    modality_statuses: Dict[str, str],
    details: Dict[str, Any],
    risk_level: str = "LOW",
) -> str:
    """Generates a responsive SVG string representing the multimodal contradiction graph."""
    # Filter active modalities
    all_keys = [
        ("image", "Facial Biometrics", "DeepFace"),
        ("voice", "Voice Acoustics", "Resemblyzer"),
        ("chat", "Chat Context", "BART Zero-Shot"),
        ("transcript", "Spoken Content", "Whisper STT"),
    ]

    active_modalities = []
    for key, label, engine in all_keys:
        status = modality_statuses.get(key, "CAN'T TELL")
        if _is_modality_active(key, status, details):
            active_modalities.append((key, label, engine, status))

    # Empty State if no evidence was provided
    if not active_modalities:
        return """
        <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 2.5rem; text-align: center; color: #64748b;">
            <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🔍</div>
            <div style="font-weight: 700; color: #1e293b; font-size: 1.05rem;">No Active Evidence Modalities Detected</div>
            <div style="font-size: 0.9rem; margin-top: 0.25rem;">Submit photos, audio clips, or incoming chat messages to map cross-modal relationships.</div>
        </div>
        """

    # Layout Geometry
    svg_width = 820
    num_nodes = len(active_modalities)

    # Calculate height and node positions deterministically
    if num_nodes <= 3:
        svg_height = 360
        id_x, id_y = svg_width / 2, 60
        # Compute horizontal spacing for children
        spacing = svg_width / (num_nodes + 1)
        node_positions = {}
        for i, (key, _, _, _) in enumerate(active_modalities):
            nx = spacing * (i + 1)
            ny = 270
            node_positions[key] = (nx, ny)
    else:
        # 4 nodes: 2x2 symmetrical layout
        svg_height = 420
        id_x, id_y = svg_width / 2, 55
        node_positions = {
            active_modalities[0][0]: (210, 195),
            active_modalities[1][0]: (610, 195),
            active_modalities[2][0]: (210, 335),
            active_modalities[3][0]: (610, 335),
        }

    # Style colors for statuses
    status_palette = {
        "PASS": {"stroke": "#10b981", "badge_bg": "#059669", "text": "AGREES", "fill": "#ecfdf5", "border": "#6ee7b7"},
        "FAIL": {"stroke": "#ef4444", "badge_bg": "#dc2626", "text": "CONTRADICTS", "fill": "#fef2f2", "border": "#fca5a5"},
        "WARNING": {"stroke": "#f59e0b", "badge_bg": "#d97706", "text": "BORDERLINE", "fill": "#fffbeb", "border": "#fcd34d"},
        "CAN'T TELL": {"stroke": "#94a3b8", "badge_bg": "#64748b", "text": "INCONCLUSIVE", "fill": "#f8fafc", "border": "#cbd5e1"},
    }

    svg_lines = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" style="width: 100%; height: auto; max-height: {svg_height}px; display: block; font-family: Inter, system-ui, -apple-system, sans-serif;">')

    # Definitions (Gradients and Markers)
    svg_lines.append("""
    <defs>
        <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%">
            <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.06"/>
        </filter>
        <marker id="arrow-green" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 8 5 L 0 9 z" fill="#10b981"/>
        </marker>
        <marker id="arrow-red" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 8 5 L 0 9 z" fill="#ef4444"/>
        </marker>
        <marker id="arrow-amber" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 8 5 L 0 9 z" fill="#f59e0b"/>
        </marker>
        <marker id="arrow-slate" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 8 5 L 0 9 z" fill="#94a3b8"/>
        </marker>
    </defs>
    """)

    # 1. Draw Edges from Claimed Identity to Active Modalities
    for key, label, engine, status in active_modalities:
        nx, ny = node_positions[key]
        pal = status_palette.get(status, status_palette["CAN'T TELL"])

        # Coordinates for connection
        start_x, start_y = id_x, id_y + 30
        end_x, end_y = nx, ny - 32

        dash = 'stroke-dasharray="6,4"' if status == "CAN'T TELL" else ""
        marker = {
            "PASS": "url(#arrow-green)",
            "FAIL": "url(#arrow-red)",
            "WARNING": "url(#arrow-amber)",
            "CAN'T TELL": "url(#arrow-slate)",
        }.get(status, "url(#arrow-slate)")

        # Smooth cubic or direct line
        svg_lines.append(f'<path d="M {start_x} {start_y} C {start_x} {(start_y + end_y)/2}, {end_x} {(start_y + end_y)/2}, {end_x} {end_y}" fill="none" stroke="{pal["stroke"]}" stroke-width="2.5" {dash} marker-end="{marker}"/>')

        # Relationship badge along the edge
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        badge_text = pal["text"]
        badge_w = 94 if len(badge_text) > 8 else 74

        svg_lines.append(f'<rect x="{mid_x - badge_w/2}" y="{mid_y - 10}" width="{badge_w}" height="20" rx="10" fill="{pal["badge_bg"]}" />')
        svg_lines.append(f'<text x="{mid_x}" y="{mid_y + 4}" fill="#ffffff" font-size="10.5" font-weight="700" text-anchor="middle" letter-spacing="0.04em">{badge_text}</text>')

    # 2. Draw Cross-Modal Edge if both Chat and Transcript are present
    if "chat" in node_positions and "transcript" in node_positions:
        cx, cy = node_positions["chat"]
        tx, ty = node_positions["transcript"]
        chat_status = modality_statuses.get("chat", "CAN'T TELL")
        tr_status = modality_statuses.get("transcript", "CAN'T TELL")

        both_pass = chat_status == "PASS" and tr_status == "PASS"
        has_fail = "FAIL" in (chat_status, tr_status)
        cross_color = "#ef4444" if has_fail else ("#10b981" if both_pass else "#f59e0b")
        cross_label = "DISCREPANCY" if has_fail else ("ALIGNED TEXT" if both_pass else "CORRELATED")

        # Horizontal connector
        svg_lines.append(f'<path d="M {cx + 105} {cy} L {tx - 105} {ty}" fill="none" stroke="{cross_color}" stroke-width="1.8" stroke-dasharray="4,3"/>')
        cmid_x = (cx + tx) / 2
        svg_lines.append(f'<rect x="{cmid_x - 48}" y="{cy - 9}" width="96" height="18" rx="4" fill="#ffffff" stroke="{cross_color}" stroke-width="1.2"/>')
        svg_lines.append(f'<text x="{cmid_x}" y="{cy + 4}" fill="{cross_color}" font-size="9.5" font-weight="700" text-anchor="middle">{cross_label}</text>')

    # 3. Draw Central Claimed Identity Node
    id_w, id_h = 240, 58
    id_risk_color = "#dc2626" if risk_level == "HIGH" else ("#d97706" if risk_level == "UNCERTAIN" else "#059669")

    clean_id = (claimed_identity or "Rahul").strip()
    if len(clean_id) > 22:
        clean_id = clean_id[:20] + "..."

    svg_lines.append(f"""
    <g transform="translate({id_x - id_w/2}, {id_y - id_h/2})" filter="url(#shadow)">
        <rect width="{id_w}" height="{id_h}" rx="8" fill="#0f172a" stroke="#334155" stroke-width="1.5"/>
        <circle cx="24" cy="{id_h/2}" r="12" fill="#1e293b"/>
        <text x="24" y="{id_h/2 + 4.5}" fill="#38bdf8" font-size="13" font-weight="800" text-anchor="middle">ID</text>
        <text x="46" y="24" fill="#f8fafc" font-size="13.5" font-weight="700">{clean_id}</text>
        <text x="46" y="42" fill="#94a3b8" font-size="10.5" font-weight="500">Claimed Identity Subject</text>
        <circle cx="{id_w - 18}" cy="{id_h/2}" r="6" fill="{id_risk_color}"/>
    </g>
    """)

    # 4. Draw Modality Evidence Nodes
    card_w, card_h = 210, 64
    for key, label, engine, status in active_modalities:
        nx, ny = node_positions[key]
        pal = status_palette.get(status, status_palette["CAN'T TELL"])

        # Fetch score detail
        score_val = ""
        if key == "image":
            dist = details.get("image", {}).get("distance")
            score_val = f"Dist: {dist:.3f}" if dist is not None else ""
        elif key == "voice":
            sim = details.get("voice", {}).get("similarity")
            score_val = f"Sim: {sim:.3f}" if sim is not None else ""
        elif key == "chat":
            cs = details.get("chat_signal", {}).get("final_score")
            score_val = f"Threat: {cs:.2f}" if cs is not None else ""
        elif key == "transcript":
            ts = details.get("transcript_signal", {}).get("final_score")
            score_val = f"Threat: {ts:.2f}" if ts is not None else ""

        svg_lines.append(f"""
        <g transform="translate({nx - card_w/2}, {ny - card_h/2})" filter="url(#shadow)">
            <rect width="{card_w}" height="{card_h}" rx="8" fill="{pal['fill']}" stroke="{pal['border']}" stroke-width="1.6"/>
            <!-- Modality Title -->
            <text x="14" y="24" fill="#0f172a" font-size="12.5" font-weight="700">{label}</text>
            <!-- Engine Subtitle -->
            <text x="14" y="42" fill="#64748b" font-size="10" font-weight="500">{engine}</text>
            <!-- Score text if available -->
            <text x="14" y="55" fill="#475569" font-size="9.5" font-weight="600">{score_val}</text>
            <!-- Status Badge in Node -->
            <rect x="{card_w - 74}" y="12" width="62" height="20" rx="10" fill="{pal['badge_bg']}"/>
            <text x="{card_w - 43}" y="26" fill="#ffffff" font-size="9.5" font-weight="800" text-anchor="middle">{status}</text>
        </g>
        """)

    svg_lines.append('</svg>')
    return "".join(svg_lines)
