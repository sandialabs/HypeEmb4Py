stars="***************************************"
dashes="---------------------------------------"
networks='friendster'
nland=100
for network in $networks
do
    printf '\n'
    echo "Computing landmarks and distances (w/ numpy) for $network..."
    mprof run -o "mprof_run_2.dat" python -u graph_preproc2_np_mprof.py $network $nland
    echo "ploting memory profile summary..."
    mprof plot -o "mprof_pproc2_np_$network.png" "mprof_run_2.dat" 
    echo "Cleaning..."
    rm "mprof_run_2.dat"
done

echo "All done!"