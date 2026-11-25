# for n_projections in 3 4 ; do for projection_method in fourier_slice real_rotate; do python joint_gradient_descent.py optimization.sliced_wasserstein.n_projections=${n_projections} optimization.sliced_wasserstein.projection_method=${projection_method} optimization.n_iterations=100; done; done

logs_to_csv() {
    echo "fname_log,projection_method,n_projections,time_per_iteration"

    for log in "$@"; do
        awk -v fname="$log" '
            /projection_method:/ {
                gsub(/^[[:space:]]*|[[:space:]]*$/, "", $2)
                projection_method = $2
            }

            /n_projections:/ {
                n_projections = $2
            }

            /Average time per iteration for/ {
                method = $0
                sub(/^.*Average time per iteration for /, "", method)
                sub(/:.*/, "", method)

                time = $0
                sub(/^.*: /, "", time)
                sub(/ seconds.*$/, "", time)
            }

            END {
                if (projection_method != "" && n_projections != "" && time != "") {
                    printf "%s,%s,%s,%s\n", fname, projection_method, n_projections, time
                } else {
                    printf "WARNING: Could not parse %s\n", fname > "/dev/stderr"
                }
            }
        ' "$log"
    done
}

logs_to_csv /mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-11-25/13-59*/joint_gradient_descent.log /mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-11-25/14-*/joint_gradient_descent.log > /mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/projection_method_scaling.csv /mnt/home/gwoollard/ceph/repos/GOTO-SWAP/examples/outputs/2026-11-25/15*/joint_gradient_descent.log