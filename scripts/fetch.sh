#!/usr/bin/env bash
# usage: scripts/fetch.sh COIN YYYYMMDD [YYYYMMDD_END]      (GNU date; run on the VPS)
# Requester-pays buckets: transfer cost is billed to your AWS account. Credentials stay on the VPS.
set -euo pipefail
COIN=$1; D=$2; E=${3:-$2}; RAW=${RAW:-data/raw}
while [[ "$D" -le "$E" ]]; do
  aws s3 cp "s3://hyperliquid-archive/market_data/$D/" "$RAW/l2/$COIN/$D/" --recursive \
    --exclude '*' --include "*/l2Book/$COIN.lz4" --request-payer requester --only-show-errors
  if [[ ! -d "$RAW/fills/$D" ]]; then
    aws s3 cp "s3://hl-mainnet-node-data/node_fills_by_block/hourly/$D/" "$RAW/fills/$D/" --recursive \
      --request-payer requester --only-show-errors
  fi
  nl=$(find "$RAW/l2/$COIN/$D" -name "$COIN.lz4" 2>/dev/null | wc -l)
  nf=$(find "$RAW/fills/$D" -type f 2>/dev/null | wc -l)
  echo "$D  l2 hours: $nl  fill files: $nf"
  if [[ "$nl" -eq 0 || "$nf" -eq 0 ]]; then
    echo "  missing -> check: aws s3 ls s3://hyperliquid-archive/market_data/$D/ --request-payer requester"
    echo "                    aws s3 ls s3://hl-mainnet-node-data/node_fills_by_block/hourly/$D/ --request-payer requester"
  fi
  D=$(date -u -d "$D + 1 day" +%Y%m%d)
done
