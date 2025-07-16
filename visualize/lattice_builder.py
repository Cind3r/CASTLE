from pyvis.network import Network
import json

def plot_pyvis_3d_lattice_interactive(molecular_tracker, layer_name, filename="Molecular_Lattice_3D_Visualization"):

    """Visualize the molecular lattice for a specific layer across all epochs in an interactive 3D format"""
    
    net = Network(height='900px', width='100%', notebook=True, directed=False)
    net.barnes_hut()

    lattice_ids = molecular_tracker.layer_lattices[layer_name]
    lattices = [molecular_tracker.lattice_library[lid] for lid in lattice_ids]
    n_epochs = len(lattices)
    n_rows = len(lattices[0].molecules)
    n_cols = len(lattices[0].molecules[0])

    all_symbols = sorted({mol.atomic_symbol for lattice in lattices for row in lattice.molecules for mol in row})
    color_map = {sym: f"hsl({int(360*i/len(all_symbols))},70%,50%)" for i, sym in enumerate(all_symbols)}

    node_ids = {}
    assembly_paths = {}
    epoch_data = {}  # Track which epoch each node belongs to
    
    for z, lattice in enumerate(lattices):
        for i, row in enumerate(lattice.molecules):
            for j, mol in enumerate(row):
                node_id = f"{z}_{i}_{j}"
                label = mol.atomic_symbol
                color = color_map[mol.atomic_symbol]
                
                # Find assembly path for this node (history of this molecule at this position)
                assembly_path = []
                for prev_z in range(z, -1, -1):
                    prev_mol = lattices[prev_z].molecules[i][j]
                    if prev_mol == mol:
                        assembly_path.append(f"{prev_z}_{i}_{j}")
                    else:
                        break
                assembly_path = assembly_path[::-1]  # earliest to latest
                
                title = (f"Molecule: {mol.atomic_symbol}"
                         f"\nWeight: {mol.atomic_weight:.4f}"
                         f"\nPosition: ({i},{j})"
                         f"\nLayer: {lattice.layer_name}"
                         f"\nEpoch: {lattice.epoch}"
                         f"\nAssembly Path: " + "".join(assembly_path))
                
                net.add_node(node_id, label=label, title=title, color=color, level=z, size=80)
                node_ids[(z, i, j)] = node_id
                assembly_paths[node_id] = assembly_path
                epoch_data[node_id] = z

    # Intra-layer bonds
    for z, lattice in enumerate(lattices):
        n_rows = len(lattice.molecules)
        n_cols = len(lattice.molecules[0])
        for i in range(n_rows):
            for j in range(n_cols):
                node_id = node_ids[(z, i, j)]
                if j < n_cols - 1:
                    net.add_edge(node_id, node_ids[(z, i, j+1)], color='gray')
                if i < n_rows - 1:
                    net.add_edge(node_id, node_ids[(z, i+1, j)], color='gray')

    # Inter-epoch assembly pathway (reuse)
    for z in range(n_epochs - 1):
        lattice_a = lattices[z]
        lattice_b = lattices[z+1]
        n_rows = min(len(lattice_a.molecules), len(lattice_b.molecules))
        n_cols = min(len(lattice_a.molecules[0]), len(lattice_b.molecules[0]))
        for i in range(n_rows):
            for j in range(n_cols):
                mol_a = lattice_a.molecules[i][j]
                mol_b = lattice_b.molecules[i][j]
                if mol_a == mol_b:
                    net.add_edge(node_ids[(z, i, j)], node_ids[(z+1, i, j)], color='blue', width=10, title='Assembly Pathway')

    # Save the network to HTML
    net.save_graph('molecular_lattice_3d_all_epochs_interactive.html')

    # Inject custom JavaScript for interactive highlighting and reset
    with open(f'docs/graphs/{filename}.html', 'r', encoding='utf-8') as f:
        html = f.read()

    # Prepare data for JS
    assembly_paths_json = json.dumps(assembly_paths)
    epoch_data_json = json.dumps(epoch_data)

    # open custom JS file
    with open('customize_plot.txt', 'r', encoding='utf-8') as jsf:
        custom_js_template = jsf.read()
    custom_js = custom_js_template.format(
        assembly_paths_json=assembly_paths_json,
        epoch_data_json=epoch_data_json
    )

    # Insert custom JS before </body>
    html = html.replace("</body>", custom_js + "\n</body>")
    with open(f'docs/graphs/{filename}.html', 'w', encoding='utf-8') as f:
        f.write(html)

    # update index.html with the new graph link
    add_graph_link_to_index(f"{filename}.html")

    print(f"Interactive 3D lattice visualization saved as 'docs/graphs/{filename}.html'.")


def add_graph_link_to_index(filename, display_name=None, index_path="index.html"):
    """
    Adds a new graph link to the index.html file.
    Args:
        filename (str): The HTML filename of the graph (e.g., "my_graph.html").
        display_name (str, optional): The text to display for the link. Defaults to the filename without extension.
        index_path (str): Path to the index.html file.
    """
    import os

    if display_name is None:
        display_name = os.path.splitext(os.path.basename(filename))[0].replace("_", " ").title()

    # Read the current index.html
    with open(index_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the <ul> and </ul> lines
    ul_start = None
    ul_end = None
    for i, line in enumerate(lines):
        if "<ul>" in line:
            ul_start = i
        if "</ul>" in line:
            ul_end = i
            break

    if ul_start is None or ul_end is None or ul_end <= ul_start:
        raise ValueError("Could not find <ul> section in index.html")

    # Prepare the new link line
    new_link = f'    <li><a href="graphs/{filename}" target="_blank">{display_name}</a></li>\n'

    # Check if the link already exists
    if new_link in lines:
        print("Link already exists in index.html")
        return

    # Insert the new link before </ul>
    lines.insert(ul_end, new_link)

    # Write back the updated file
    with open(index_path, "w", encoding="utf-8") as f:
        f.writelines(lines)