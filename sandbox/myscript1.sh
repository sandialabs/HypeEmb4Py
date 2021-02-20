stars="***************************************"
dashes="---------------------------------------"
networks='friendster'
nland=100
for network in $networks
do
    printf '\n'
    echo "Computing landmarks and distances for $network..."
    mprof run -o "mprof_run_1.dat" python -u graph_preproc2_mprof.py $network $nland
    echo "ploting memory profile summary..."
    mprof plot -o "mprof_pproc2_$network.png" "mprof_run_1.dat"
    echo "Cleaning..."
    rm "mprof_run_1.dat"
done

echo "All done!"