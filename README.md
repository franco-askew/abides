# ABIDES: Agent-Based Interactive Discrete Event Simulation environment
### For HIP-3 Perpetual Futures
Agent-based discrete event simulation of a Hyperliquid HIP-3 builder-deployed perpetual futures market, built on the [ABIDES](https://arxiv.org/pdf/1904.12066) framework.

The platform simulates a perpetual contract with an untradeable underlying -- the only external input is an oracle price fed via CSV. All market logic (matching, margin, funding, liquidation, mark price, fees) replicates HIP-3 semantics. Agents interact through the ABIDES message-passing kernel with configurable pairwise latencies.

## Quickstart

```
git clone <repo-url>
cd abides
pip install -r requirements.txt

# Run with the sample oracle (5-minute sim, no trading agents)
python abides.py -c hip3_perp -- --oracle-csv data/sample_oracle.csv \
    --start-time "2025-01-01 00:00:00" --end-time "2025-01-01 00:05:00"

# Run with trading agents
python abides.py -c hip3_perp -- --oracle-csv data/sample_oracle.csv \
    --start-time "2025-01-01 00:00:00" --end-time "2025-01-01 00:05:00" \
    --num-agents 5 --log-orders

# Run the end-to-end integration test
python tests/test_perp_e2e.py
```

See the [wiki](https://github.com/franco-askew/abides-hip-3/wiki) for documentation.
