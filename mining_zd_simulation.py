import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from classes import ExactZDStrategy, MiningSimulation

# Global parameters
block_reward = 6.25  # BTC
elec_cost_factor = 0.1
learning_rate = 0.05
kappa = 0.1
sabotage_bonus = 0.1
skimming_rate = 0.03
scale_factor = 100
migration_rate = 0.01


def plot_zd_strategy_analysis():
    """Analyse ZD strategies mathematically"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Generous ZD strategies
    axes[0].set_title('a) Generous ZD Strategies (χ variation)')
    chi_values = [1.5, 2.0, 3.0, 5.0]
    colors = ["#FF6B6B", "#E7B67B", "#75C6D7", "#7FDA73"]
    # Calculate proper bar positions
    num_chis = len(chi_values)
    bar_width = 0.18
    outcome_positions = np.arange(4)  # 4 outcomes: CC, CD, DC, DD

    for idx, (chi, color) in enumerate(zip(chi_values, colors)):
        zd = ExactZDStrategy(strategy_type='generous', chi=chi)
        p_vec = zd.get_p_vector()

        offset = (idx - num_chis/2 + 0.5) * bar_width
        x = outcome_positions + offset
        axes[0].bar(x, p_vec, width=bar_width, label=f'χ={chi}',
                    color=color, alpha=0.8, edgecolor='black')

    axes[0].set_xlabel('Outcome (CC, CD, DC, DD)')
    axes[0].set_ylabel('Probability of Cooperation')
    axes[0].set_xticks(outcome_positions)
    axes[0].set_xticklabels(['CC', 'CD', 'DC', 'DD'])
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, linestyle='--')

    # Bitcoin Mining Economics
    axes[1].set_title('b) Bitcoin Mining Payoff Structure')
    B = block_reward
    f = 0.025  # 2.5% fee
    alpha = 0.01
    c_e = elec_cost_factor * alpha * B
    mining_payoffs = {
        'CC': ((1-f)*alpha*B - c_e, f*alpha*B),
        'CD': ((1-f)*alpha*B - c_e + sabotage_bonus*alpha*B, -(1-f)*alpha*B),
        'DC': ((1-f-skimming_rate)*alpha*B - c_e, (f+skimming_rate)*alpha*B),
        'DD': (-c_e, 0)
    }

    outcomes = list(mining_payoffs.keys())
    miner_pay = [mining_payoffs[o][0] * scale_factor for o in outcomes]
    pool_pay = [mining_payoffs[o][1] * scale_factor for o in outcomes]
    x = np.arange(len(outcomes))
    width = 0.35
    axes[1].bar(x - width/2, miner_pay, width, label='Miner',
                color='#4ECDC4', edgecolor='black')
    axes[1].bar(x + width/2, pool_pay, width, label='Pool',
                color='#FF6B6B', edgecolor='black')
    axes[1].set_xlabel('Game Outcome')
    axes[1].set_ylabel('Payoff (scaled)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(outcomes)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.show()

    bfig, baxes = plt.subplots(1, 2, figsize=(12, 6))
    # ZD Relationship Verification
    baxes[0].set_title('a) ZD Payoff Relationship Verification')
    # Test generous ZD against different opponents
    zd = ExactZDStrategy(strategy_type='generous', chi=2.0)
    opponents = ['AllC', 'AllD', 'TFT', 'Random']
    opponent_q = {
        'AllC': np.array([1.0, 1.0, 1.0, 1.0]),
        'AllD': np.array([0.0, 0.0, 0.0, 0.0]),
        'TFT': np.array([1.0, 0.0, 1.0, 0.0]),
        'Random': np.array([0.5, 0.5, 0.5, 0.5]),
    }
    opponent_payoffs = []
    zd_payoffs = []

    for opp_name in opponents:
        q = opponent_q[opp_name]
        p = zd.get_p_vector()
        # Calculate stationary distribution
        M = np.array([
            [p[0]*q[0], p[0]*(1-q[0]), (1-p[0])*q[0], (1-p[0])*(1-q[0])],
            [p[1]*q[1], p[1]*(1-q[1]), (1-p[1])*q[1], (1-p[1])*(1-q[1])],
            [p[2]*q[2], p[2]*(1-q[2]), (1-p[2])*q[2], (1-p[2])*(1-q[2])],
            [p[3]*q[3], p[3]*(1-q[3]), (1-p[3])*q[3], (1-p[3])*(1-q[3])]
        ])

        eigenvalues, eigenvectors = np.linalg.eig(M.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = np.abs(stationary / np.sum(stationary))
        # Calculate payoffs
        R, S, T, P = 3.0, 0.0, 5.0, 1.0
        S_X = np.array([R, S, T, P])
        S_Y = np.array([R, T, S, P])

        s_X = np.dot(stationary, S_X)
        s_Y = np.dot(stationary, S_Y)

        opponent_payoffs.append(s_Y)
        zd_payoffs.append(s_X)

        baxes[0].scatter(s_Y, s_X, label=opp_name, s=100)

    # Plot ZD relationship line: s_X - R = χ(s_Y - R)
    x_vals = np.linspace(1, 5, 100)
    y_vals = 3.0 + 2.0 * (x_vals - 3.0)
    baxes[0].plot(x_vals, y_vals, color='grey', linestyle='--',
                  label=f's_X - R = {zd.chi}·(s_Y - R)')

    baxes[0].set_xlabel('Opponent Payoff (s_Y)')
    baxes[0].set_ylabel('ZD Player Payoff (s_X)')
    baxes[0].legend()
    baxes[0].grid(True, alpha=0.3, linestyle='--')

    # Strategy Space
    baxes[1].set_title('b) Strategy Space in Prisoner\'s Dilemma')
    # Plot feasible payoff region
    R, S, T, P = 3.0, 0.0, 5.0, 1.0
    feasible_points = []

    # Generate random strategies to show feasible region
    for _ in range(1000):
        p = np.random.random(4)  # Random memory-one strategy
        q = np.random.random(4)

        M = np.array([
            [p[0]*q[0], p[0]*(1-q[0]), (1-p[0])*q[0], (1-p[0])*(1-q[0])],
            [p[1]*q[1], p[1]*(1-q[1]), (1-p[1])*q[1], (1-p[1])*(1-q[1])],
            [p[2]*q[2], p[2]*(1-q[2]), (1-p[2])*q[2], (1-p[2])*(1-q[2])],
            [p[3]*q[3], p[3]*(1-q[3]), (1-p[3])*q[3], (1-p[3])*(1-q[3])]
        ])

        eigenvalues, eigenvectors = np.linalg.eig(M.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = np.abs(stationary / np.sum(stationary))

        s_X = np.dot(stationary, [R, S, T, P])
        s_Y = np.dot(stationary, [R, T, S, P])

        feasible_points.append((s_X, s_Y))

    feasible_points = np.array(feasible_points)
    baxes[1].scatter(feasible_points[:, 1], feasible_points[:, 0],
                     alpha=0.1, s=10, color='gray', label='Feasible region')

    # Mark special points
    special_strategies = {
        'Mutual Cooperation (CC)': (R, R),
        'Mutual Defection (DD)': (P, P),
        'Miner Defects (CD)': (S, T),
        'Pool Defects (DC)': (T, S),
    }

    for name, (x, y) in special_strategies.items():
        baxes[1].scatter(y, x, s=100, label=name, edgecolor='black')

    baxes[1].set_xlabel('Miner Payoff (s_Y)')
    baxes[1].set_ylabel('Pool Payoff (s_X)')
    baxes[1].legend(fontsize=8)
    baxes[1].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()


def plot_simulation_results(sim, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=14)

    # Cooperation Rate Over Time
    axes[0, 0].plot(sim.history_coop_rate, color="#3CA3CF", linewidth=2)
    axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('a) Cooperation Rate Evolution')
    axes[0, 0].set_xlabel('Round', fontsize=10)
    axes[0, 0].set_ylabel('Cooperation Rate', fontsize=10)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')

    # Strategy Evolution
    axes[0, 1].set_title('b) Strategy Evolution')
    strategies = set()
    for dist in sim.history_strategy_distribution:
        strategies.update(dist.keys())

    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
    for idx, strat in enumerate(sorted(strategies)):
        counts = [dist.get(strat, 0)
                  for dist in sim.history_strategy_distribution]
        axes[0, 1].plot(counts, label=strat, linewidth=2, color=colors[idx])

    axes[0, 1].set_xlabel('Round', fontsize=10)
    axes[0, 1].set_ylabel('Number of Miners', fontsize=10)
    axes[0, 1].set_ylim(0, 70)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')

    # Average Miner Payoff
    # Create a simple network visualisation
    G = sim.network
    pos = nx.spring_layout(G, seed=42)
    axes[1, 1].set_title('d) Final Network Graph')

    # Color nodes by strategy
    node_colors = []
    for i in range(len(sim.miners)):
        miner = sim.miners[i]
        if miner.strategy_type == 'Cooperator':
            node_colors.append("#4ECD83")
        elif miner.strategy_type == 'Defector':
            node_colors.append('#FF6B6B')
        elif 'ZD' in miner.strategy_type:
            node_colors.append('#FFD166')
        else:
            node_colors.append("#A8DADC")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=axes[1, 1],
                           node_size=50, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=axes[1, 1], alpha=0.1)
    axes[1, 1].axis('off')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#4ECD83",
                   markersize=10, label='Cooperator'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B',
                   markersize=10, label='Defector'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD166',
                   markersize=10, label='ZD Strategy'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#A8DADC',
                   markersize=10, label='Other'),
    ]
    axes[1, 1].legend(handles=legend_elements, loc='upper left', fontsize=8)

    # Pool Payoffs
    axes[1, 0].set_title('c) Pool Cumulative Payoffs')
    for pool in sim.pools:
        payoffs = [p.get(pool.id, 0) for p in sim.history_pool_payoffs]
        axes[1, 0].plot(payoffs, label=pool.strategy_type, linewidth=2)

    axes[1, 0].set_xlabel('Round', fontsize=10)
    axes[1, 0].set_ylabel('Cumulative Payoff', fontsize=10)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set seed for consistency
    SEED = 42  # Use a constant
    random.seed(SEED)
    np.random.seed(SEED)

    print("BITCOIN MINING WITH EXACT ZERO-DETERMINANT STRATEGIES")
    print("\nMathematical Analysis of ZD Strategies...")

    # First, show the mathematical analysis
    plot_zd_strategy_analysis()

    # RQ1: Single Generous ZD Pool
    print("RESEARCH QUESTION 1: Can Generous ZD Suppress Attacks?")

    sim1 = MiningSimulation(num_miners=100, rounds=1000,
                            network_type="barabasi_albert")
    sim1.setup_scenario_rq1()
    sim1.run()

    plot_simulation_results(sim1, "Single Generous ZD Pool Performance")

    # RQ2
    print("RESEARCH QUESTION 2: Network Topology Effects")
    networks = ["barabasi_albert", "erdos_renyi", "watts_strogatz"]
    topology_results = {}
    topology_sims = {}

    for net in networks:
        print(f"\nTesting {net} network...")
        sim = MiningSimulation(num_miners=100, rounds=800, network_type=net)
        sim.setup_scenario_rq1()
        sim.run()
        topology_results[net] = sim.history_coop_rate[-1]
        topology_sims[net] = sim

    # Plot network comparison with graphs
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # Top row: Network graphs
    network_colors = ['#FF9999', '#66B2FF', '#99FF99']
    network_titles = {
        'barabasi_albert': 'Barabási-Albert',
        'erdos_renyi': 'Erdős-Rényi',
        'watts_strogatz': 'Watts-Strogatz'
    }

    for idx, (net, color) in enumerate(zip(networks, network_colors)):
        ax = fig.add_subplot(gs[0, idx])
        sim = topology_sims[net]
        G = sim.network

        # Use spring layout with fixed seed for consistency
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        # Color nodes by strategy
        node_colors = []
        for i in range(len(sim.miners)):
            miner = sim.miners[i]
            if miner.strategy_type == 'Cooperator':
                node_colors.append("#4ECD83")
            elif miner.strategy_type == 'Defector':
                node_colors.append('#FF6B6B')
            elif 'ZD' in miner.strategy_type:
                node_colors.append('#FFD166')
            else:
                node_colors.append("#A8DADC")

        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=50, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5, ax=ax)

        # Add network statistics
        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
        clustering = nx.average_clustering(G)

        ax.set_title(f'a) {network_titles[net]}\nAvg Degree: {avg_degree:.2f},'
                     f' Clustering: {clustering:.3f}',
                     fontsize=12, fontstyle='italic')
        ax.axis('off')

    # Add legend for node colors
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#4ECD83",
                   markersize=8, label='Cooperator'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B',
                   markersize=8, label='Defector'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD166',
                   markersize=8, label='ZD Strategy'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#A8DADC',
                   markersize=10, label='Other'),
    ]
    fig.legend(handles=legend_elements, loc='center',
               ncol=4, bbox_to_anchor=(0.5, 0.52), fontsize=8, frameon=True)

    # Bottom row: Cooperation rates and bar chart
    ax_coop = fig.add_subplot(gs[1, :2])
    for net, color in zip(networks, network_colors):
        sim = topology_sims[net]
        ax_coop.plot(sim.history_coop_rate,
                     label=network_titles[net].replace('\n', ' '),
                     linewidth=2, color=color)

    ax_coop.set_title('b) Cooperation Rate Evolution by Network Type',
                      fontsize=12)
    ax_coop.set_xlabel('Round', fontsize=10)
    ax_coop.set_ylabel('Cooperation Rate', fontsize=10)
    ax_coop.set_ylim(0, 1.1)
    ax_coop.legend(loc='best', fontsize=8)
    ax_coop.grid(True, alpha=0.3, linestyle='--')
    ax_coop.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Bottom right: Final cooperation bar chart
    ax_bar = fig.add_subplot(gs[1, 2])
    network_names = [network_titles[n].replace('\n', ' ') for n in networks]
    coop_rates = [topology_results[net] for net in networks]

    bars = ax_bar.bar(range(len(networks)), coop_rates, color=network_colors,
                      edgecolor='black', width=0.6)
    ax_bar.set_title('c) Final Cooperation Rate', fontsize=12)
    ax_bar.set_ylabel('Cooperation Rate', fontsize=10)
    ax_bar.set_xticks(range(len(networks)))
    ax_bar.set_xticklabels(['Scale-Free', 'Random', 'Small-World'],
                           rotation=15, ha='right', fontsize=10)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', fontsize=10)

    fig.suptitle('RQ2: Network Topology Effects on Cooperation',
                 fontsize=14, y=0.98)
    plt.show()

    # RQ3: Pool Competition
    print("RESEARCH QUESTION 3: Pool Strategy Competition")

    sim3 = MiningSimulation(num_miners=150, rounds=1000,
                            network_type="watts_strogatz")
    sim3.setup_scenario_rq2_comparative()
    sim3.run()

    # Plot pool competition results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('RQ3: Pool Strategy Competition Results', fontsize=14)

    # Market Share
    pools = [f"{p.strategy_type}\n({len(p.members)})" for p in sim3.pools]
    members = [len(p.members) for p in sim3.pools]
    colors = ["#FF6B6B", "#E7B67B", "#75C6D7", "#7FDA73"][:len(pools)]

    axes[0].pie(members, labels=pools, colors=colors, autopct='%1.1f%%',
                startangle=90, wedgeprops={'edgecolor': 'black'})
    axes[0].set_title('a) Final Market Share')

    # Profitability
    payoffs = [p.cumulative_payoff for p in sim3.pools]
    x_pos = np.arange(len(pools))
    bars = axes[1].bar(x_pos, payoffs, color=colors, edgecolor='black')
    axes[1].set_title('b) Total Pool Profit')
    axes[1].set_ylabel('Cumulative Payoff', fontsize=10)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([p.strategy_type for p in sim3.pools], rotation=45,
                            fontsize=10)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     height + (100 if height > 0 else -100),
                     f'{height:.0f}', ha='center',
                     va='bottom' if height > 0 else 'top')

    # Cooperation Rate
    axes[2].plot(sim3.history_coop_rate, color="#3FA1CB", linewidth=2)
    axes[2].set_title('c) Cooperation Rate in Competition')
    axes[2].set_xlabel('Round', fontsize=10)
    axes[2].set_ylabel('Cooperation Rate', fontsize=10)
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()
