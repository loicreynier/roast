#!/usr/bin/env sh

HOSTNAME="${HOSTNAME:-$(hostname)}"

module purge
case "$HOSTNAME" in
    *pmcs2i.ec-lyon.fr)
    module load Python/3.9.5-GCCcore-10.3.0
        ;;
    *)
        echo "\$HOSTNAME unknown, no module loaded"
esac
