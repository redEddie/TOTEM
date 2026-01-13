python lg3/prepare_lg3_data.py \
  --freq 15min \
  --smooth_window 3 \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --ereport_cols "Capa_Cooling,MFR_068,Rop,Comp1 Hz_1,Comp1 Hz_0,VAP_Entha,LIQ_Entha,cycle,HighP,LowP,Tcond,SCEEV_M" \
  --smartcare_cols "Tset,Tid,Hid,Low P,High P,Power,Tpip_in,Frun" \
  --smartcare_mode per_unit \
  --exclude_time_range "00:00-06:00"
