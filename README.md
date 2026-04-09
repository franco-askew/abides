# ABIDES: Agent-Based Interactive Discrete Event Simulation environment
### For HIP-3 Perpetual Futures
Agent-based discrete event simulation of a Hyperliquid HIP-3 builder-deployed perpetual futures market, built on the [ABIDES](https://arxiv.org/pdf/1904.12066) framework.

The platform simulates a perpetual contract with an untradeable underlying -- the only external input is an oracle price fed via CSV. All market logic (matching, margin, funding, liquidation, mark price, fees) replicates HIP-3 semantics. Agents interact through the ABIDES message-passing kernel with configurable pairwise latencies.
