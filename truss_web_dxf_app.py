from pathlib import Path
import zipfile

# Streamlit app code that reads a DXF file and performs truss analysis
dxf_web_app_code = '''
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import ezdxf
from io import BytesIO

st.set_page_config(layout="wide")
st.title("DXF-Based Truss Analysis Web App")

st.sidebar.header("1. Upload DXF File")
dxf_file = st.sidebar.file_uploader("Upload a DXF file", type=["dxf"])

if dxf_file:
    # Read DXF file
    doc = ezdxf.read(stream=dxf_file)
    msp = doc.modelspace()

    # Extract unique nodes and lines
    node_coords = []
    member_lines = []

    def find_or_add_node(pt):
        for i, coord in enumerate(node_coords):
            if np.allclose(coord, pt, atol=1e-6):
                return i
        node_coords.append(pt)
        return len(node_coords) - 1

    for e in msp:
        if e.dxftype() == 'LINE':
            p1 = (e.dxf.start.x, e.dxf.start.y)
            p2 = (e.dxf.end.x, e.dxf.end.y)
            idx1 = find_or_add_node(p1)
            idx2 = find_or_add_node(p2)
            member_lines.append((idx1, idx2))

    st.success(f"Loaded {len(node_coords)} nodes and {len(member_lines)} members from DXF.")

    # Display truss diagram with numbered nodes and members
    st.subheader("2. Truss Diagram")
    fig, ax = plt.subplots()
    for i, (i1, i2) in enumerate(member_lines):
        x1, y1 = node_coords[i1]
        x2, y2 = node_coords[i2]
        ax.plot([x1, x2], [y1, y2], 'k-')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, f"M{i+1}", color='blue', fontsize=8)
    for i, (x, y) in enumerate(node_coords):
        ax.plot(x, y, 'ro')
        ax.text(x, y, f"N{i+1}", fontsize=8, ha='right')
    ax.set_aspect('equal')
    ax.set_title("Truss Shape from DXF")
    st.pyplot(fig)

    # Input table for cross-sectional area and Young's modulus
    st.subheader("3. Material Properties Input")
    member_data = []
    for i in range(len(member_lines)):
        cols = st.columns(3)
        cols[0].write(f"Member M{i+1}")
        A = cols[1].number_input(f"A{i}", label_visibility="collapsed", value=1.0, format="%.4f", key=f"A_{i}")
        E = cols[2].number_input(f"E{i}", label_visibility="collapsed", value=29000.0, format="%.1f", key=f"E_{i}")
        member_data.append((A, E))

    if st.button("Run Analysis"):
        # Step 1: Assemble nodes and members
        nodes = [{'index': i, 'x': x, 'y': y} for i, (x, y) in enumerate(node_coords)]
        members = []
        for i, ((n1, n2), (A, E)) in enumerate(zip(member_lines, member_data)):
            dx = nodes[n2]['x'] - nodes[n1]['x']
            dy = nodes[n2]['y'] - nodes[n1]['y']
            l = np.hypot(dx, dy)
            t = np.arctan2(dy, dx)
            members.append({'index': i, 'n1': n1, 'n2': n2, 'A': A, 'E': E, 'l': l, 't': t})

        # Assume all nodes are "free" and zero loads for simplicity
        dof_map = [True for _ in range(len(nodes)*2)]
        F = np.zeros(len(nodes)*2)

        # Step 2: Global stiffness matrix
        dof = 2 * len(nodes)
        Kg = np.zeros((dof, dof))
        for mem in members:
            c, s = np.cos(mem['t']), np.sin(mem['t'])
            k = (mem['A'] * mem['E']) / mem['l']
            C = np.array([
                [ c**2,  c*s, -c**2, -c*s],
                [ c*s,  s**2, -c*s, -s**2],
                [-c**2, -c*s,  c**2,  c*s],
                [-c*s, -s**2,  c*s,  s**2]
            ])
            K_local = k * C
            i, j = mem['n1'], mem['n2']
            idx = [2*i, 2*i+1, 2*j, 2*j+1]
            for a in range(4):
                for b in range(4):
                    Kg[idx[a], idx[b]] += K_local[a, b]

        indices = [i for i, dof in enumerate(dof_map) if dof]
        Kg_r = Kg[np.ix_(indices, indices)]
        F_r = F[indices]
        Q_r = np.linalg.solve(Kg_r, F_r)
        Q_full = np.zeros(len(dof_map))
        idx = 0
        for i, dof in enumerate(dof_map):
            if dof:
                Q_full[i] = Q_r[idx]
                idx += 1

        # Step 3: Stresses
        stresses = []
        for mem in members:
            c, s = np.cos(mem['t']), np.sin(mem['t'])
            i, j = mem['n1'], mem['n2']
            d = np.array([Q_full[2*i], Q_full[2*i+1], Q_full[2*j], Q_full[2*j+1]])
            B = np.array([-c, -s, c, s])
            stress = (mem['E'] / mem['l']) * B.dot(d)
            stresses.append(stress)

        # Step 4: Results
        st.subheader("4. Results")
        st.write("### Global Stiffness Matrix")
        st.dataframe(Kg)

        st.write("### Node Displacements")
        for i in range(len(nodes)):
            st.write(f"Node {i+1}: dx = {Q_full[2*i]:.6f}, dy = {Q_full[2*i+1]:.6f}")

        st.write("### Member Stresses")
        for i, s in enumerate(stresses):
            st.write(f"Member M{i+1}: {s:.6f} ksi")

        st.write("### Truss with Displacements and Stresses")
        fig, ax = plt.subplots()
        for mem, stress in zip(members, stresses):
            x1, y1 = nodes[mem['n1']]['x'], nodes[mem['n1']]['y']
            x2, y2 = nodes[mem['n2']]['x'], nodes[mem['n2']]['y']
            ax.plot([x1, x2], [y1, y2], color='red' if stress > 0 else 'blue', linewidth=2)
            ax.text((x1+x2)/2, (y1+y2)/2, f"{stress:.2f}", fontsize=8, ha='center')
        for i, node in enumerate(nodes):
            dx, dy = Q_full[2*i], Q_full[2*i+1]
            ax.plot(node['x'], node['y'], 'ko')
            ax.text(node['x'], node['y'], f"N{i+1}\\n({dx:.4f},{dy:.4f})", fontsize=8)
        ax.set_aspect('equal')
        ax.set_title("Truss Analysis Result")
        ax.set_xlabel("X (in)")
        ax.set_ylabel("Y (in)")
        ax.grid(True)
        st.pyplot(fig)