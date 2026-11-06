# Phase 0.5 Documentation: Data Ingestion & Ground Truth Setup

This document records the exact steps, dataset details, tools executed, and outputs produced during Phase 0.5 to prepare the network capture data and ground-truth labels for the Kitsune NIDS vs. Polymorphic WGAN evaluation pipeline.

---

## 1. Directory Structure Created
Organized storage according to [AGENTS.md](file:///home/brooklyng3/projects/grad/AGENTS.md) and [.agentignore](file:///home/brooklyng3/projects/grad/.agentignore):
- `data/raw/` — Holds the original PCAP, ground truth labels, and raw parsed packet headers (TSV).
- `data/processed/` — Reserved for extracted statistical feature matrices and intermediate tensors.
- `data/attack_samples/` — Reserved for isolated malicious samples and perturbed adversarial outputs.

---

## 2. Raw Traffic Ingestion (`mirai.pcap`)
- **Source:** Extracted from the `mirai.zip` archive bundled with `ymirsky/Kitsune-py`.
- **Target Destination:** `data/raw/mirai.pcap`
- **File Metadata:**
  - Size: 60 MB uncompressed (62,859,800 bytes)
  - Total Packet Count: 764,137 packets (verified via `capinfos`)
  - Capture Type: Ethernet / IP / TCP / UDP / ICMP / ARP traffic tracking Mirai IoT botnet propagation and scanning.

---

## 3. Ground Truth Labels Acquisition (`mirai_labels.csv`)
- **Source:** Official Kitsune Network Attack Dataset benchmark repository (mirrored via `zyyrrr/attack-defense`).
- **Target Destination:** `data/raw/mirai_labels.csv`
- **File Metadata:**
  - Size: ~2.3 MB (2,292,409 bytes)
  - Total Label Lines: 764,136 binary indicators
  - Label Encoding:
    - `0` = Benign / normal IoT network traffic
    - `1` = Malicious (Mirai Telnet brute-force infection and scanning)
  - Alignment: 1-to-1 sequential alignment with packets in `mirai.pcap` (excluding the TSV header row).

---

## 4. Native `tshark` PCAP Dissection (`mirai.pcap.tsv`)
To bypass Scapy's performance bottleneck and ensure rapid streaming ingestion, `tshark` (v4.2.2) was executed to parse all packet header fields into a tab-delimited intermediate file matching Kitsune's `FeatureExtractor.py` specification:

- **Command Executed:**
  ```bash
  /usr/bin/tshark -r data/raw/mirai.pcap -T fields \
    -e frame.time_epoch \
    -e frame.len \
    -e eth.src \
    -e eth.dst \
    -e ip.src \
    -e ip.dst \
    -e tcp.srcport \
    -e tcp.dstport \
    -e udp.srcport \
    -e udp.dstport \
    -e icmp.type \
    -e icmp.code \
    -e arp.opcode \
    -e arp.src.hw_mac \
    -e arp.src.proto_ipv4 \
    -e arp.dst.hw_mac \
    -e arp.dst.proto_ipv4 \
    -e ipv6.src \
    -e ipv6.dst \
    -E header=y -E occurrence=f > data/raw/mirai.pcap.tsv
  ```
- **Execution Performance:**
  - Runtime: ~1 minute 12 seconds
  - CPU: Single core utilized during dissection
  - Memory: Streamed with minimal RAM footprint (< 250 MB)
- **Output File Metadata:**
  - Location: `data/raw/mirai.pcap.tsv`
  - Size: 94 MB
  - Total Lines: 764,138 (1 header line + 764,137 packet rows)

---

## 5. File & Invariant Integrity Check
- **Label / Packet Count Validation:**
  - `data/raw/mirai.pcap` = 764,137 packets
  - `data/raw/mirai.pcap.tsv` = 764,137 data rows (+ 1 header)
  - `data/raw/mirai_labels.csv` = 764,137 labels (Note: `wc -l` reported 764,136 because the final line has no trailing newline `\n`. Exact line/entry count parsed via Python is exactly 764,137, achieving 100% 1-to-1 match).
- **Ignore Discipline:** All created data files (`mirai.pcap`, `mirai.pcap.tsv`, `mirai_labels.csv`) match [.agentignore](file:///home/brooklyng3/projects/grad/.agentignore) and [.gitignore](file:///home/brooklyng3/projects/grad/.gitignore) patterns and will only be accessed via buffered stream iterators.
