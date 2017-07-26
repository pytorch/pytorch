# Latency optimization

These tips apply to Linux.

### NUMA locality

The process calling into Gloo algorithms should ideally be pinned to a
single NUMA node. If it isn't, the scheduler can decide to move back
and forth between nodes, which typically hurts performance.

With different NUMA nodes representing different PCIe root complexes,
make sure that the NUMA node you run on is the same one that the NIC
you use is connected to. Requiring transfers from and to the NIC to
traverse root complexes means introducing additional latency and
unnecessary inter-processor communication.

It is not enough to only pin the process to a particular NUMA node
(e.g. using `numactl(8)`). The NIC being used needs to have ALL its
interrupts pinned to this NUMA node as well. You can verify this by
looking at `/proc/interrupts` and configure this through
`/proc/irq/${IRQ}/smp_affinity`. See the [documentation on SMP IRQ
affinity][100] for more information.

[100]: https://www.kernel.org/doc/Documentation/IRQ-affinity.txt

### TCP tuning

In no particular order:

#### Enable TSO (TCP segmentation offload)

Make sure it is enabled if your NIC supports it. For high bandwidth
NICs, this is absolutely necessary to achieve line rate on a single
connection (some anecdotal evindence: 10Gb/s without TSO at 100% CPU
usage versus 40Gb/s (line rate) with TSO at 30% CPU usage).

```
# ethtool -k eth0 | grep segmentation
tcp-segmentation-offload: on
        tx-tcp-segmentation: on
        tx-tcp6-segmentation: on
```

#### Disable ER (Early Retransmit) and TLP (Tail Loss Probe)

Uses valuable kernel cycles and not needed in network environments
where Gloo is typically used (low latency, packet drop extremely
rare). ER and TLP are configured using the same sysctl and both can be
disabled.

```
echo 0 > /proc/sys/net/ipv4/tcp_early_retrans
```

For more information, see [`ip-sysctl.txt`][200] (see
`tcp_early_retrans`).

[200]: https://www.kernel.org/doc/Documentation/networking/ip-sysctl.txt
