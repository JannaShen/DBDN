#######################################################
###### No self-ensemble: DBDN
# BI degradation model, X2, X3, X4
th Test.lua -model DBDN_x2.t7 -degradation BI -scale 2 -selfEnsemble false -dataset Set5
th Test.lua -model DBDN_x3.t7 -degradation BI -scale 3 -selfEnsemble false -dataset Set5
th Test.lua -model DBDN_x4.t7 -degradation BI -scale 4 -selfEnsemble false -dataset Set5


###### With self-ensemble: DBDN+
# BI degradation model, X2, X3, X4
th Test.lua -model DBDNplus_x2.t7 -degradation BI -scale 2 -selfEnsemble true -dataset Set5
th Test.lua -model DBDNplus_x3.t7  -degradation BI -scale 3 -selfEnsemble true -dataset Set5
th Test.lua -model DBDNplus_x4.t7  -degradation BI -scale 4 -selfEnsemble true -dataset Set5









