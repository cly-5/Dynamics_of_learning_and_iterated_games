import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# Define Standard IPD Payoffs
T = 5  # Temptation
R = 3  # Reward
P = 1  # Punishment
S = 0  # Sucker


if __name__ == "__main__":
    # Define the 4 states
    states = ['CC', 'CD', 'DC', 'DD']

    # Create a directed graph
    G = nx.DiGraph()
    G.add_nodes_from(states)

    # Define edges (transitions)
    # We add all possible transitions.
    # In a static diagram, we usually just show the structure,
    # not every single probability label (too messy).
    for s1 in states:
        for s2 in states:
            G.add_edge(s1, s2)

    # Position the nodes in a square
    pos = {
        'CC': (0, 1),
        'CD': (1, 1),
        'DC': (0, 0),
        'DD': (1, 0)
    }

    plt.figure(figsize=(8, 6))

    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000,
                           node_color='white', edgecolors='black')

    # Draw Labels
    nx.draw_networkx_labels(G, pos)

    # Draw Edges (Curved arrows)
    # Self-loops (staying in same state)
    for state in states:
        nx.draw_networkx_edges(G, pos, edgelist=[(state, state)],
                               connectionstyle='arc3, rad=0.5',
                               arrowstyle='->', arrowsize=20)

    # Cross transitions
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[('CC', 'CD'), ('CD', 'DD'), ('DD', 'DC'), ('DC', 'CC')],
        connectionstyle='arc3, rad=0.2', arrowstyle='->', arrowsize=15)

    nx.draw_networkx_edges(
        G, pos,
        edgelist=[('CD', 'CC'), ('DD', 'CD'), ('DC', 'DD'), ('CC', 'DC')],
        connectionstyle='arc3, rad=0.2', arrowstyle='->',
        arrowsize=15, style='dashed')

    # Diagonals
    nx.draw_networkx_edges(G, pos, edgelist=[('CC', 'DD'), ('DD', 'CC')],
                           connectionstyle='arc3, rad=0.1', arrowstyle='->',
                           arrowsize=15, alpha=0.5)

    nx.draw_networkx_edges(G, pos, edgelist=[('CD', 'DC'), ('DC', 'CD')],
                           connectionstyle='arc3, rad=0.1', arrowstyle='->',
                           arrowsize=15, alpha=0.5)

    plt.title("State Space of the Memory-One IPD", pad=20, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Define the polygon of feasible payoffs
    # Corners: (P,P), (S,T), (R,R), (T,S)
    hull_x = [P, S, R, T, P]
    hull_y = [P, T, R, S, P]

    # Fill the feasible region
    ax.fill(hull_x, hull_y, color='lightgray',
            alpha=0.3, label='Feasible Payoff Set')
    ax.plot(hull_x, hull_y, 'k-', alpha=0.7)

    # Plot the 4 Static Outcomes
    outcomes = {
        '(P,P)': (P, P),
        '(S,T)': (S, T),
        '(R,R)': (R, R),
        '(T,S)': (T, S)
    }

    for label, (x, y) in outcomes.items():
        ax.plot(x, y, 'ko', markersize=8)
        # Offset label slightly
        ax.text(x + 0.1, y + 0.1, label)

    # Exact Extortion Strategy (Chi = 3, Baseline P)
    chi_ext = 3.0
    # Calculate intersection of line and boundary
    y_ext_limit = 10.5 / 5.5  # Exact intersection y ~ 1.909

    y_vals_ext = np.linspace(P, y_ext_limit, 100)
    x_vals_ext = chi_ext * (y_vals_ext - P) + P

    ax.plot(x_vals_ext, y_vals_ext, 'r-', linewidth=2,
            label=f'Extortion ZD ($\\chi={chi_ext}$, Base $P$)')

    # Exact Generous Strategy (Chi = 2, Baseline R)
    chi_gen = 2.0
    # Calculate the intersection of line and boundary
    y_gen_limit = 17 / 9  # Exact intersection y ~ 1.888
    y_vals_gen = np.linspace(y_gen_limit, R, 100)
    x_vals_gen = chi_gen * (y_vals_gen - R) + R

    ax.plot(x_vals_gen, y_vals_gen, 'g-', linewidth=2,
            label=f'Generous ZD ($\\chi={chi_gen}$, Base $R$)')

    # Plot Tit-for-Tat
    ax.plot([P, R], [P, R], 'b--', alpha=0.6, label='Tit-for-Tat')

    # Formatting
    ax.set_title("Zero-Determinant Strategy Linear Relationships", fontsize=14)
    ax.set_xlabel(r"Player X Payoff ($s_X$)", fontsize=12)
    ax.set_ylabel(r"Player Y Payoff ($s_Y$)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set limits clearly
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 5.5)

    plt.tight_layout()
    plt.show()
