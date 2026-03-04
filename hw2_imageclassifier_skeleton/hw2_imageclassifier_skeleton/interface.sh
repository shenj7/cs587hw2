#
DEVICE=cuda
EPOCHS=50


#
cnn=0
shuffle=0
amp=0
eqcnn=0
for arg in $*; do
    #
    if [[ ${arg} == "cnn" ]]; then
        #
        cnn=1
    elif [[ ${arg} == "shuffle" ]]; then
        #
        shuffle=1
    elif [[ ${arg} == "amp" ]]; then
        #
        amp=1
    elif [[ ${arg} == "eqcnn" ]]; then
        #
        eqcnn=1
    else
        #
        echo "unknown interface ${arg}."
        exit 1
    fi
done

EPOCHS=100
#
if [[ ${cnn} -gt 0 ]]; then
    # Q3.c.1 and Q3.c.2: CNN with different kernel/stride
    sbatch scholar.sh python main.py --batch-size 100 --cnn --lr 1e-3 \
        --kernel 5 --stride 1 --device ${DEVICE} --num-epochs ${EPOCHS}
    sbatch scholar.sh python main.py --batch-size 100 --cnn --lr 1e-3 \
        --kernel 3 --stride 3 --device ${DEVICE} --num-epochs ${EPOCHS}
    sbatch scholar.sh python main.py --batch-size 100 --cnn --lr 1e-3 \
        --kernel 14 --stride 1 --device ${DEVICE} --num-epochs ${EPOCHS}
fi

#
if [[ ${shuffle} -gt 0 ]]; then
    # Q3.c.3: Shuffle labels experiment
    sbatch scholar.sh python main.py --batch-size 100 --cnn --lr 1e-2 \
        --shuffle-label --device ${DEVICE} --num-epochs ${EPOCHS}
fi


EPOCHS=50
#
if [[ ${eqcnn} -gt 0 ]]; then
    # Q3.c.4: Equivariant CNN experiments (trains CNN baseline + equivariant CNN)
    sbatch scholar.sh python train_eqcnn.py --device ${DEVICE} --epochs ${EPOCHS}
fi
