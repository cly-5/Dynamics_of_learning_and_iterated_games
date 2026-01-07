import numpy as np
import networkx as nx
import random

# Global parameters
block_reward = 6.25  # BTC
elec_cost_factor = 0.1
learning_rate = 0.05
kappa = 0.1
sabotage_bonus = 0.1
skimming_rate = 0.03
scale_factor = 100
migration_rate = 0.01


class ExactZDStrategy:
    def __init__(self, strategy_type="generous", chi=2.0):
        self.strategy_type = strategy_type
        self.chi = chi
        # Canonical Prisoner's Dilemma payoffs for ZD construction
        self.R = 3.0
        self.S = 0.0
        self.T = 5.0
        self.P = 1.0
        # Calculate exact p-vector
        self.p_vector = self._calculate_exact_zd_vector()
        # Verify the ZD relationship holds
        self.zd_error = self._verify_zd_relationship()
        if self.zd_error > 1e-10:
            print(f"Note: ZD strategy approximate error = {self.zd_error:.6f}")

    def _calculate_exact_zd_vector(self):
        """Calculate exact ZD strategy vector using Press-Dyson equations"""
        R, S, T, P = self.R, self.S, self.T, self.P
        chi = self.chi

        if self.strategy_type == 'generous':
            # Target: S_X - R = chi * (S_Y - R)
            baseline = np.array([1.0, 1.0, 0.0, 0.0])
            M = np.array([
                0,                          # CC: Balanced at (R,R)
                (S - R) - chi * (T - R),    # CD: Negative value
                (T - R) - chi * (S - R),    # DC: Positive value
                (P - R) - chi * (P - R)     # DD: Positive value
            ])

        elif self.strategy_type == 'extortion':
            # Target: S_X - P = chi * (S_Y - P)
            baseline = np.array([1.0, 1.0, 0.0, 0.0])
            M = np.array([
                (R - P) - chi * (R - P),    # CC: Negative value
                (S - P) - chi * (T - P),    # CD: Negative value
                (T - P) - chi * (S - P),    # DC: Positive value
                0                           # DD: Balanced at (P,P)
            ])

        else:  # Equaliser or other
            return np.array([1.0, 0.5, 0.5, 0.0])

        phi_candidates = []

        for i in range(4):
            if M[i] == 0:
                continue

            if M[i] > 0:
                # baseline + phi * M <= 1  =>  phi <= (1 - baseline) / M
                limit = (1.0 - baseline[i]) / M[i]
                if limit > 0:  # Only consider positive phi
                    phi_candidates.append(limit)

            elif M[i] < 0:
                # baseline + phi * M >= 0  =>  phi <= -baseline / M
                limit = -baseline[i] / M[i]
                if limit > 0:
                    phi_candidates.append(limit)

        # Select the smallest valid phi to satisfy all constraints
        if not phi_candidates:
            phi = 0.0
        else:
            phi = min(phi_candidates) * 0.9999

        # Compute Final Vector
        p_vector = baseline + phi * M
        # Ensure strict bounds [0, 1]
        return np.clip(p_vector, 0.0, 1.0)

    def _verify_zd_relationship(self):
        """Verify ZD relationship holds against standard opponents"""
        test_opponents = {
            'AlwaysCooperate': np.array([1.0, 1.0, 1.0, 1.0]),
            'AlwaysDefect': np.array([0.0, 0.0, 0.0, 0.0]),
            'TitForTat': np.array([1.0, 0.0, 1.0, 0.0]),
            'Random': np.array([0.5, 0.5, 0.5, 0.5]),
        }
        errs = []
        for opp_name, q in test_opponents.items():
            p = self.p_vector
            # Build 4x4 Markov transition matrix
            M = np.array([
                [p[0]*q[0], p[0]*(1-q[0]), (1-p[0])*q[0], (1-p[0])*(1-q[0])],
                [p[1]*q[1], p[1]*(1-q[1]), (1-p[1])*q[1], (1-p[1])*(1-q[1])],
                [p[2]*q[2], p[2]*(1-q[2]), (1-p[2])*q[2], (1-p[2])*(1-q[2])],
                [p[3]*q[3], p[3]*(1-q[3]), (1-p[3])*q[3], (1-p[3])*(1-q[3])]
            ])

            # Find stationary distribution using Power Method
            v = np.ones(4) / 4
            for _ in range(200):  # Iterate to convergence
                v_next = np.dot(v, M)
                v = v_next / np.sum(v_next)

            stationary = v
            # Calculate expected payoffs, order: CC, CD, DC, DD
            S_X = np.array([self.R, self.S, self.T, self.P])
            S_Y = np.array([self.R, self.T, self.S, self.P])

            s_X = np.dot(stationary, S_X)
            s_Y = np.dot(stationary, S_Y)

            # Check linear constraint based on strategy type
            if self.strategy_type == 'generous':
                # Expected: s_X - R = chi * (s_Y - R)
                lhs = s_X - self.R
                rhs = self.chi * (s_Y - self.R)
            elif self.strategy_type == 'extortion':
                # Expected: s_X - P = chi * (s_Y - P)
                lhs = s_X - self.P
                rhs = self.chi * (s_Y - self.P)
            else:
                lhs, rhs = 0, 0

            errs.append(np.abs(lhs - rhs))

        # Return the maximum error found across all opponents
        return max(errs) if errs else 0.0

    def decide_action(self, last_outcome):
        """Make decision based on last outcome"""
        if last_outcome is None:
            return 'C'  # First move: cooperate
        outcome_to_idx = {'CC': 0, 'CD': 1, 'DC': 2, 'DD': 3}
        idx = outcome_to_idx[last_outcome]
        p_cooperate = self.p_vector[idx]

        return 'C' if random.random() < p_cooperate else 'D'

    def get_p_vector(self):
        """Return the strategy vector"""
        return self.p_vector.copy()

    def get_zd_relationship(self):
        """Return the ZD relationship this strategy enforces"""
        if self.strategy_type == 'generous':
            return f"s_X - {self.R} = {self.chi}·(s_Y - {self.R})"
        elif self.strategy_type == 'extortion':
            return f"s_X - {self.P} = {self.chi}·(s_Y - {self.P})"
        return "Unknown Relationship"


class MinerAgent:
    def __init__(self, node_id, hashrate, strategy_type="TitForTat"):
        self.id = node_id
        self.hashrate = hashrate
        self.strategy_type = strategy_type
        self.pool = None
        self.cumulative_payoff = 0

        # Initialise ZD strategy if needed
        if strategy_type.startswith("ZDGen"):
            self.zd_strategy = ExactZDStrategy(
                strategy_type='generous',
                chi=float(strategy_type[-1]) if strategy_type[-1].isdigit()
                else 2.0)
            self.has_zd = True

        elif strategy_type.startswith("ZDExt"):
            self.zd_strategy = ExactZDStrategy(
                strategy_type='extortion',
                chi=float(strategy_type[-1]) if strategy_type[-1].isdigit()
                else 2.0)
            self.has_zd = True

        else:
            self.zd_strategy = None
            self.has_zd = False

        # Memory for non-ZD strategies
        self.last_opponent_action = None

    def decide_action(self, last_outcome):
        """Decide action based on strategy"""
        if self.has_zd:
            return self.zd_strategy.decide_action(last_outcome)  # ZD strategy
        if last_outcome:
            self.last_opponent_action = last_outcome[0]  # Non-ZD strategies
        if self.strategy_type == "Cooperator":
            return 'C'
        elif self.strategy_type == "Defector":
            return 'D'
        elif self.strategy_type == "TitForTat":
            if self.last_opponent_action is None:
                return 'C'
            return self.last_opponent_action
        elif self.strategy_type == "Random":
            return random.choice(['C', 'D'])
        elif self.strategy_type == "Grudger":
            if self.last_opponent_action == 'D':
                return 'D'  # Never forgive
            return 'C'
        elif self.strategy_type == "GTFT":
            if self.last_opponent_action == 'D':
                return 'C' if random.random() < 0.1 else 'D'  # 10% forgiveness
            return 'C'
        else:
            return 'C'

    def update_strategy(self, neighbor, temperature=0.1):
        """Fermi rule for strategy adoption"""
        if self.has_zd and neighbor.has_zd:
            # Both are ZD strategies - keep own
            return

        pi_self = self.cumulative_payoff
        pi_neighbor = neighbor.cumulative_payoff

        # Fermi probability
        diff = (pi_self - pi_neighbor) / temperature
        diff = np.clip(diff, -50, 50)  # Prevent overflow
        prob = 1 / (1 + np.exp(diff))

        if random.random() < prob:
            # Adopt neighbor's strategy
            self.strategy_type = neighbor.strategy_type
            self.has_zd = neighbor.has_zd
            if neighbor.has_zd:
                self.zd_strategy = neighbor.zd_strategy
            else:
                self.zd_strategy = None


class PoolAgent:
    def __init__(self, pool_id, fee=0.025, strategy_type="ZDGen2"):
        self.id = pool_id
        self.fee = fee
        self.strategy_type = strategy_type
        self.members = []
        self.cumulative_payoff = 0
        self.cum_mem_payoff = 0

        # Initialise strategy
        if strategy_type.startswith("ZDGen"):
            is_digit = strategy_type[-1].isdigit()
            chi = float(strategy_type[-1]) if is_digit else 2.0
            self.zd_strategy = ExactZDStrategy(strategy_type='generous',
                                               chi=chi)
            self.has_zd = True

        elif strategy_type.startswith("ZDExt"):
            if strategy_type[-1].isdigit():
                chi = float(strategy_type[-1])
            else:
                chi = 2.0
            self.zd_strategy = ExactZDStrategy(strategy_type='extortion',
                                               chi=chi)
            self.has_zd = True

        else:
            self.zd_strategy = None
            self.has_zd = False

    def decide_payout_action(self, last_outcome):
        """Decide payout action based on strategy"""
        if self.has_zd:
            return self.zd_strategy.decide_action(last_outcome)

        # Non-ZD pool strategies
        if self.strategy_type == "TitForTat":
            if last_outcome is None:
                return 'C'
            return last_outcome[1] if last_outcome else 'C'
        elif self.strategy_type == "Cooperator":
            return 'C'
        elif self.strategy_type == "Defector":
            return 'D'
        elif self.strategy_type == "GTFT":
            if last_outcome and last_outcome[1] == 'D':
                return 'C' if random.random() < 0.1 else 'D'
            return 'C'
        else:
            return 'C'


class MiningSimulation:
    def __init__(self, num_miners=100, rounds=500,
                 network_type="barabasi_albert"):
        self.num_miners = num_miners
        self.rounds = rounds
        self.miners = []
        self.pools = []
        self.network = self._create_network(network_type, num_miners)
        self.last_outcomes = {}
        self.history_coop_rate = []
        self.history_strategy_distribution = []
        self.history_pool_payoffs = []
        self.history_avg_miner_payoff = []

    def _create_network(self, network_type, n):
        """Create network topology"""
        networks = {
            "barabasi_albert": lambda: nx.barabasi_albert_graph(n, 2),
            "erdos_renyi": lambda: nx.erdos_renyi_graph(n, 0.05),
            "watts_strogatz": lambda: nx.watts_strogatz_graph(n, 4, 0.1),
        }
        return networks.get(network_type, networks["barabasi_albert"])()

    def setup_scenario_rq1(self):
        """RQ1: Single Generous ZD pool"""
        pool = PoolAgent(0, fee=0.025, strategy_type="ZDGen2")
        self.pools.append(pool)

        # Strategy distribution
        strategy_distribution = {
            "Cooperator": 0.30,
            "Defector": 0.30,
            "TitForTat": 0.20,
            "Random": 0.10,
            "Grudger": 0.05,
            "ZDGen2": 0.05,
        }

        strategies = []
        for strat, proportion in strategy_distribution.items():
            count = int(self.num_miners * proportion)
            strategies.extend([strat] * count)

        # Fill any remaining slots
        while len(strategies) < self.num_miners:
            strategies.append("TitForTat")

        random.shuffle(strategies)

        for i in range(self.num_miners):
            hashrate = np.random.pareto(a=2.0) + 0.1
            strategy = strategies[i]

            miner = MinerAgent(i, hashrate, strategy_type=strategy)
            miner.pool = pool
            pool.members.append(miner)
            self.miners.append(miner)

        print(f"Setup RQ1: {len(self.miners)} miners, 1 ZDGen2 pool")
        print(f"Pool fee: {pool.fee*100:.1f}%")
        if pool.has_zd:
            print(f"ZD Strategy Vector: {pool.zd_strategy.get_p_vector()}")
            print(f"ZD Relationship: {pool.zd_strategy.get_zd_relationship()}")

    def setup_scenario_rq2_comparative(self):
        """RQ2: Multiple pools with different strategies"""
        pool_configs = [
            ("ZDGen2", 0.025),    # Generous ZD
            ("TitForTat", 0.025),  # Standard TFT
            ("ZDGen3", 0.020),    # More generous ZD
            ("GTFT", 0.030),      # Generous TFT
        ]

        for idx, (strat, fee) in enumerate(pool_configs):
            pool = PoolAgent(idx, fee=fee, strategy_type=strat)
            self.pools.append(pool)

        # Distribute miners
        strategies = [
            "Cooperator", "Defector", "TitForTat", "Random", "ZDGen2"
            ]
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]

        for i in range(self.num_miners):
            hashrate = np.random.pareto(a=2.0) + 0.1
            strategy = random.choices(strategies, weights=weights)[0]

            miner = MinerAgent(i, hashrate, strategy_type=strategy)

            # Choose pool based on performance (simplified)
            if len(self.pools) > 0:
                pool = random.choice(self.pools)
                miner.pool = pool
                pool.members.append(miner)

            self.miners.append(miner)

        print(f"Setup RQ2: {len(self.miners)} miners, {len(self.pools)} pools")

    def calculate_payoff(self, pool_action, miner_action, alpha, fee):
        """Calculate Bitcoin mining payoffs"""
        B = block_reward
        c_e = elec_cost_factor * alpha * B

        state = f"{pool_action}{miner_action}"

        if state == 'CC':  # Mutual cooperation
            miner_pay = (1 - fee) * alpha * B - c_e
            pool_pay = fee * alpha * B
        elif state == 'CD':  # Miner withholds
            miner_pay = (1 - fee) * alpha * B - c_e + sabotage_bonus*alpha*B
            pool_pay = -(1 - fee) * alpha * B  # Pool pays but gets nothing
        elif state == 'DC':  # Pool skims
            miner_pay = (1 - fee - skimming_rate) * alpha * B - c_e
            pool_pay = (fee + skimming_rate) * alpha * B
        else:  # DD - Mutual defection
            miner_pay = -c_e
            pool_pay = 0

        return miner_pay * scale_factor, pool_pay * scale_factor

    def run(self):
        """Execute simulation"""
        print(f"\nStarting Simulation: {self.num_miners} miners, "
              f"{self.rounds} rounds")
        print(f"Network: {type(self.network).__name__}")
        print(f"Pools: {[p.strategy_type for p in self.pools]}")
        print("Running...\n")

        for t in range(self.rounds):
            round_coops = 0
            total_actions = 0
            round_payoffs = []

            # Interaction Phase
            for miner in self.miners:
                pool = miner.pool
                if not pool:
                    continue
                # Get last outcome for this pair
                key = (miner.id, pool.id)
                last_outcome = self.last_outcomes.get(key)
                # Get actions
                miner_action = miner.decide_action(last_outcome)
                pool_action = pool.decide_payout_action(last_outcome)
                # Store new outcome for next round
                new_outcome = f"{pool_action}{miner_action}"
                self.last_outcomes[key] = new_outcome
                # Calculate payoffs
                total_hash = sum(m.hashrate for m in pool.members)
                if total_hash == 0:
                    continue

                alpha = miner.hashrate / total_hash
                fee = pool.fee
                miner_pay, pool_pay = self.calculate_payoff(
                    pool_action,  miner_action, alpha, fee
                    )

                # Update cumulative payoffs
                miner.cumulative_payoff += miner_pay
                pool.cumulative_payoff += pool_pay
                pool.cum_mem_payoff += miner_pay
                round_payoffs.append(miner_pay)
                total_actions += 1
                if miner_action == 'C' and pool_action == 'C':
                    round_coops += 1

            # Social Learning Phase (every 10 rounds)
            if t > 0 and t % 10 == 0:
                for miner in self.miners:
                    if random.random() < learning_rate:
                        neighbors = list(self.network.neighbors(miner.id))
                        if neighbors:
                            neighbor = self.miners[random.choice(neighbors)]
                            miner.update_strategy(neighbor, temperature=kappa)

            # Pool Migration Phase (for RQ3, every 50 rounds)
            if len(self.pools) > 1 and t > 0 and t % 50 == 0:
                self._pool_migration_step(t)

            # Record Metrics
            coop_rate = round_coops / max(1, total_actions)
            self.history_coop_rate.append(coop_rate)

            avg_pay = np.mean(round_payoffs) if round_payoffs else 0
            self.history_avg_miner_payoff.append(avg_pay)

            # Record strategy distribution
            strat_dist = {}
            for miner in self.miners:
                strat = miner.strategy_type
                strat_dist[strat] = strat_dist.get(strat, 0) + 1
            self.history_strategy_distribution.append(strat_dist)

            # Record pool payoffs
            pool_pays = {p.id: p.cumulative_payoff for p in self.pools}
            self.history_pool_payoffs.append(pool_pays)

            if t % 100 == 0:
                print(f"Round {t:4d}: Coop={coop_rate:.3f}, "
                      f"AvgPayoff={avg_pay:6.3f}")

        print("\n Simulation Complete!")
        self._print_summary()

    def _pool_migration_step(self, t):
        """Allow miners to switch pools based on performance"""
        if len(self.pools) <= 1:
            return

        for miner in self.miners:
            if random.random() < migration_rate and miner.pool:
                current_pool = miner.pool
                # Calculate miner's current average payoff (rate payoff/round)
                current_avg = miner.cumulative_payoff / (t + 1) if t > 0 else 0
                # Find best alternative pool
                best_alt = None
                best_score = -float('inf')

                for pool in self.pools:
                    if pool != current_pool and len(pool.members) > 0:
                        # Metric: Average payoff per member per round
                        member_cum_avg = pool.cum_mem_payoff/len(pool.members)
                        # We divide by (t + 1) to match the units
                        pool_yield_rate = member_cum_avg / (t + 1)

                        if pool_yield_rate > best_score:
                            best_score = pool_yield_rate
                            best_alt = pool

                # Switch if alternative is significantly better (15% threshold)
                if best_alt and best_score > current_avg * 1.15:
                    current_pool.members.remove(miner)
                    best_alt.members.append(miner)
                    miner.pool = best_alt
                    # Clear history with old pool
                    key = (miner.id, current_pool.id)
                    if key in self.last_outcomes:
                        del self.last_outcomes[key]

    def _print_summary(self):
        """Print final statistics"""
        print("\n" + "="*70)
        print("RESULTS SUMMARY")

        print(f"\nFinal Cooperation Rate: {self.history_coop_rate[-1]:.3f}")
        avg_coop = np.mean(self.history_coop_rate)
        print(f"Average Cooperation Rate: {avg_coop:.3f}")

        final_strats = self.history_strategy_distribution[-1]
        print("\nFinal Strategy Distribution:")
        for strat, count in sorted(final_strats.items(), key=lambda x: -x[1]):
            pct = 100 * count / self.num_miners
            print(f"  {strat:20s}: {count:3d} miners ({pct:5.1f}%)")

        print("\nPool Performance:")
        for pool in self.pools:
            if pool.members:
                avg = pool.cumulative_payoff / len(pool.members)
                is_profit = pool.cumulative_payoff > 0
                profit_status = "PROFIT" if is_profit else "LOSS"
                print(f"  Pool {pool.id} ({pool.strategy_type:15s}): "
                      f"Total={pool.cumulative_payoff:8.2f}, "
                      f"Avg={avg:6.2f}, Members={len(pool.members):3d} "
                      f"[{profit_status}]")
