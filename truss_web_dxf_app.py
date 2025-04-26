import streamlit as st
import ezdxf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Interactive Truss Analysis from DXF")

def extract_segments(doc):
    msp = doc.modelspace()
    segs = []
    for e in msp.query('LINE LWPOLYLINE POLYLINE'):
        t = e.dxftype()
        if t == 'LINE':
            segs.append(((e.dxf.start.x, e.dxf.start.y),
                         (e.dxf.end.x,   e.dxf.end.y)))
        elif t == 'LWPOLYLINE':
            pts = [(p[0], p[1]) for p in e.get_points('xy')]
            for a, b in zip(pts, pts[1:]):
                segs.append((a, b))
            if e.closed and len(pts) > 2:
                segs.append((pts[-1], pts[0]))
        elif t == 'POLYLINE' and e.is_2d_polyline:
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            for a, b in zip(pts, pts[1:]):
                segs.append((a, b))
            if e.is_closed and len(pts) > 2:
                segs.append((pts[-1], pts[0]))
    return segs

def build_nodes_members(segments):
    unique = {}
    nodes = []
    for p, q in segments:
        for pt in (p, q):
            if pt not in unique:
                unique[pt] = len(nodes)
                nodes.append(pt)
    members = []
    for i, (p, q) in enumerate(segments):
        members.append({'index': i, 'n1': unique[p], 'n2': unique[q]})
    return nodes, members

def draw_truss(nodes, members, title="Truss Geometry"):
    fig, ax = plt.subplots(figsize=(6,6))
    for mem in members:
        x1, y1 = nodes[mem['n1']]
        x2, y2 = nodes[mem['n2']]
        ax.plot([x1, x2], [y1, y2], '-k')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, str(mem['index']), color='blue')
    for i, (x, y) in enumerate(nodes):
        ax.plot(x, y, 'ro')
        ax.text(x, y, str(i), color='red', va='bottom', ha='right')
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    ax.set_title(title)
    return fig

uploaded = st.file_uploader("1) Upload your DXF file", type="dxf")
if not uploaded:
    st.info("Please upload a .dxf to get started.")
    st.stop()

try:
    doc = ezdxf.readfile(uploaded)
except Exception as err:
    st.error(f"Failed to read DXF: {err}")
    st.stop()

segments = extract_segments(doc)
if not segments:
    st.error("No straight-line segments found in the DXF.")
    st.stop()

nodes, members = build_nodes_members(segments)
st.success(f"Found {len(nodes)} nodes and {len(members)} members.")

st.subheader("2) Truss Geometry")
fig_geo = draw_truss(nodes, members)
st.pyplot(fig_geo)

st.subheader("3) Define Node Constraints & Loads")
with st.form("node_form"):
    node_inputs = []
    for i, (x, y) in enumerate(nodes):
        st.markdown(f"**Node {i}:** ({x:.3f}, {y:.3f})")
        c = st.selectbox(f" Constraint at Node {i}", ["free","roll_x","roll_y","fixed"], key=f"c{i}")
        fx = st.number_input(f" Fx at Node {i} (kips)", value=0.0, format="%.3f", key=f"fx{i}")
        fy = st.number_input(f" Fy at Node {i} (kips)", value=0.0, format="%.3f", key=f"fy{i}")
        node_inputs.append({'constraint': c, 'fx': fx, 'fy': fy})
    submitted_nodes = st.form_submit_button("Save Node Data")

if not submitted_nodes:
    st.info("Fill out all node data and click **Save Node Data**.")
    st.stop()

st.subheader("4) Define Member Areas & Moduli")
with st.form("member_form"):
    mem_inputs = []
    for mem in members:
        idx = mem['index']
        n1, n2 = mem['n1'], mem['n2']
        st.markdown(f"**Member {idx}: Node {n1} → Node {n2}**")
        A = st.number_input(f" Area A (in²) for Member {idx}", value=1.0, format="%.4f", key=f"A{idx}")
        E = st.number_input(f" Modulus E (ksi) for Member {idx}", value=29000.0, format="%.1f", key=f"E{idx}")
        mem_inputs.append({'A': A, 'E': E})
    submitted_mems = st.form_submit_button("Save Member Data")

if not submitted_mems:
    st.info("Fill out all member data and click **Save Member Data**.")
    st.stop()

st.subheader("5) Running Analysis…")
dof_map = []
for nd in node_inputs:
    c = nd['constraint']
    if c == "free":
        dof_map += [True, True]
    elif c == "roll_x":
        dof_map += [True, False]
    elif c == "roll_y":
        dof_map += [False, True]
    else:
        dof_map += [False, False]

n = len(nodes)
dof = 2 * n
Kg = np.zeros((dof, dof))
for mem, md in zip(members, mem_inputs):
    i, j = mem['n1'], mem['n2']
    x1, y1 = nodes[i]; x2, y2 = nodes[j]
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    c, s = dx / L, dy / L
    k_local = (md['A'] * md['E'] / L) * np.array([
        [c*c,  c*s, -c*c, -c*s],
        [c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s,  c*c,  c*s],
        [-c*s, -s*s,  c*s,  s*s]
    ])
    idx_map = [2*i, 2*i+1, 2*j, 2*j+1]
    for a in range(4):
        for b in range(4):
            Kg[idx_map[a], idx_map[b]] += k_local[a, b]

F = np.zeros(dof)
for i, nd in enumerate(node_inputs):
    F[2*i] = nd['fx']
    F[2*i+1] = nd['fy']

free_idx = [i for i, ok in enumerate(dof_map) if ok]
Kg_r = Kg[np.ix_(free_idx, free_idx)]
F_r = F[free_idx]
Q_r = np.linalg.solve(Kg_r, F_r)

Q = np.zeros(dof)
cnt = 0
for i, ok in enumerate(dof_map):
    if ok:
        Q[i] = Q_r[cnt]
        cnt += 1

stresses = []
for mem, md in zip(members, mem_inputs):
    i, j = mem['n1'], mem['n2']
    x1, y1 = nodes[i]; x2, y2 = nodes[j]
    L = np.hypot(x2 - x1, y2 - y1)
    c, s = (x2 - x1) / L, (y2 - y1) / L
    d_vec = np.array([Q[2*i], Q[2*i+1], Q[2*j], Q[2*j+1]])
    B = np.array([-c, -s, c, s])
    stresses.append((md['E'] / L) * B.dot(d_vec))

st.subheader("6) Results")
disp_df = pd.DataFrame({
    "Node": list(range(n)),
    "dx (in)": Q[0::2],
    "dy (in)": Q[1::2]
})
st.table(disp_df)

stress_df = pd.DataFrame({
    "Member": [m['index'] for m in members],
    "Stress (ksi)": stresses
})
st.table(stress_df)

st.subheader("7) Deformed Shape")
scale = st.slider("Deformation Scale Factor", 1, 500, 50)
fig_def, ax = plt.subplots(figsize=(6,6))
for mem in members:
    i, j = mem['n1'], mem['n2']
    x1, y1 = nodes[i]; x2, y2 = nodes[j]
    u1x, u1y = Q[2*i] * scale, Q[2*i+1] * scale
    u2x, u2y = Q[2*j] * scale, Q[2*j+1] * scale
    ax.plot([x1+u1x, x2+u2x], [y1+u1y, y2+u2y], '-r')
ax.set_aspect('equal', 'box'); ax.grid(True)
st.pyplot(fig_def)
