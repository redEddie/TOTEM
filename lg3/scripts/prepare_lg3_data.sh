EREPORT_COLS="Capa_Cooling,MFR_068,Rop,Comp1 Hz_1,Comp1 Hz_0,VAP_Entha,LIQ_Entha,cycle,HighP,LowP,Tcond,SCEEV_M"
SMARTCARE_COLS="Tod"

python lg3/prepare_lg3_data.py \
  --freq 15min \
  --output_dir lg3/data/processed \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --ereport_cols "${EREPORT_COLS}" \
  --smartcare_process_cols "${SMARTCARE_COLS}"
