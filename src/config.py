class Stacking_MODEL:
    mlp_models=[
        "mlp_model_index1",
        "mlp_model_index2",
        "mlp_model_index3",
        "mlp_model_index4",
        "mlp_model_index5",
    ]

    lgb_models=[
        "lgb_cv_num_leaves12_seed0_index1",
        "lgb_cv_num_leaves12_seed0_index2",
        "lgb_cv_num_leaves12_seed0_index3",
        "lgb_cv_num_leaves12_seed0_index4",
        "lgb_cv_num_leaves12_seed0_index5",
        
        #"lgb_cv_num_leaves12_seed0_index6",
        #"lgb_cv_num_leaves12_seed0_index7",
        #"lgb_cv_num_leaves12_seed0_index8",
        #"lgb_cv_num_leaves12_seed0_index9",
        #"lgb_cv_num_leaves12_seed0_index10",
    ]

    lgb_models2=[
        "lgb_cv_num_leaves12_seed42_index1",
        "lgb_cv_num_leaves12_seed42_index2",
        "lgb_cv_num_leaves12_seed42_index3",
        "lgb_cv_num_leaves12_seed42_index4",
        "lgb_cv_num_leaves12_seed42_index5",
        "lgb_cv_num_leaves12_seed42_index6",
        "lgb_cv_num_leaves12_seed42_index7",
        "lgb_cv_num_leaves12_seed42_index8",
        #"lgb_cv_num_leaves12_seed42_index9",
        #"lgb_cv_num_leaves12_seed42_index10",
    ]

    lgb_models3=[
        "lgb_cv_num_leaves12_seed71_index1",
        "lgb_cv_num_leaves12_seed71_index2",
        "lgb_cv_num_leaves12_seed71_index3",
        "lgb_cv_num_leaves12_seed71_index4",
        "lgb_cv_num_leaves12_seed71_index5",
        "lgb_cv_num_leaves12_seed71_index6",
        "lgb_cv_num_leaves12_seed71_index7",
        "lgb_cv_num_leaves12_seed71_index8",
        #"lgb_cv_num_leaves12_seed71_index9",
        #"lgb_cv_num_leaves12_seed71_index10",
    ]

    rf_models=[
        "rf_max_depth5_model_index1",
        "rf_max_depth5_model_index2",
        "rf_max_depth5_model_index3",
        "rf_max_depth5_model_index4",
        "rf_max_depth5_model_index5",
        "rf_max_depth5_model_index6",
        "rf_max_depth5_model_index7",
        "rf_max_depth5_model_index8",
        "rf_max_depth5_model_index9",
        "rf_max_depth5_model_index10",
    ]

    knn_models=[
        #'single_knn_nneighbors3_model'
        #'cv_knn_nneighbors3_index1',
        #'cv_knn_nneighbors3_index2',
        #'cv_knn_nneighbors3_index3',
        #'cv_knn_nneighbors3_index4',
        #'cv_knn_nneighbors3_index5',
        #'cv_knn_nneighbors5_index6',
        #'cv_knn_nneighbors5_index7',
        #'cv_knn_nneighbors5_index8',
        #'cv_knn_nneighbors5_index9',
        #'cv_knn_nneighbors5_index10',
    ]

    ridge_models=[
        'ridge_model_index1',
        'ridge_model_index2',
        'ridge_model_index3',
        'ridge_model_index4',
        'ridge_model_index5', 
        'ridge_model_index6', 
        'ridge_model_index7', 
        'ridge_model_index8', 
        'ridge_model_index9',
        'ridge_model_index10',  
    ]

    xgb_models=[
        "xgb_model_seed0_index1",
        "xgb_model_seed0_index2",
        "xgb_model_seed0_index3",
        "xgb_model_seed0_index4",
        "xgb_model_seed0_index5",
        #"xgb_model_seed0_index6",
        #"xgb_model_seed0_index7",
        #"xgb_model_seed0_index8",
        #"xgb_model_seed0_index9",
        #"xgb_model_seed0_index10",
    ]

    xgb_models2=[
        "xgb_model_seed42_index1",
        "xgb_model_seed42_index2",
        "xgb_model_seed42_index3",
        "xgb_model_seed42_index4",
        "xgb_model_seed42_index5",
        #"xgb_model_seed42_index6",
        #"xgb_model_seed42_index7",
        #"xgb_model_seed42_index8",
        #"xgb_model_seed42_index9",
        #"xgb_model_seed42_index10",
        
    ]

    xgb_models3=[
        "xgb_model_seed71_index1",
        "xgb_model_seed71_index2",
        "xgb_model_seed71_index3",
        "xgb_model_seed71_index4",
        "xgb_model_seed71_index5",
        #"xgb_model_seed71_index6",
        #"xgb_model_seed71_index7",
        #"xgb_model_seed71_index8",
        #"xgb_model_seed71_index9",
        #"xgb_model_seed71_index10",
        
    ]



class Config:
    LGB_SPLITS=8
    MLP_SPLITS=5
    XGB_SPLITS=5
    RIDGE_SPLITS=5
    RFR_SPLITS=10
    NGB_SPLITS=5
    """
    seed値を変更しても特徴量の重要度は統一する必要がある
    1. seed=xで全データを用いて学習、重要度の算出
    2. seed=xで重要度がある閾値以上の特徴量を用いて学習
    3. seed=yで2. と同様に学習
    """
    SEED=42
    #FEATURE_SELECTION_THRESHOLD=950
    FEATURE_SELECTION_THRESHOLD=950
    USING_ALL_DATA=False
    NUM_LEAVES=12
    svr_degree=1

    """log1p(target)についての情報"""
    TARGET_MEAN = 5.696034457842397
    TARGET_STD = 2.0097708208223373

    rf_max_depth=5
    rf_n_estimators=100

    '''同一名称の特徴量がrdkit由来のデータに含まれるため削除対象(ignore_3D=TrueのFalseの場合)'''
    mordred_ignore3D_False_drop_column_list = [
    "BalabanJ","BertzCT","LabuteASA","PEOE_VSA1","PEOE_VSA10","PEOE_VSA11","PEOE_VSA12","PEOE_VSA13","PEOE_VSA2","PEOE_VSA3","PEOE_VSA4",
    "PEOE_VSA5","PEOE_VSA6","PEOE_VSA7","PEOE_VSA8","PEOE_VSA9","SMR_VSA1","SMR_VSA2","SMR_VSA3","SMR_VSA4","SMR_VSA5","SMR_VSA6","SMR_VSA7",
    "SMR_VSA8","SMR_VSA9","SlogP_VSA1","SlogP_VSA10","SlogP_VSA11","SlogP_VSA2","SlogP_VSA3","SlogP_VSA4","SlogP_VSA5","SlogP_VSA6",
    "SlogP_VSA7","SlogP_VSA8","SlogP_VSA9","TPSA","EState_VSA1","EState_VSA10","EState_VSA2","EState_VSA3","EState_VSA4","EState_VSA5",
    "EState_VSA6","EState_VSA7","EState_VSA8","EState_VSA9","VSA_EState1","VSA_EState2","VSA_EState3","VSA_EState4","VSA_EState5",
    "VSA_EState6","VSA_EState7","VSA_EState8","VSA_EState9"
    ]
    '''同一名称の特徴量がrdkit由来のデータに含まれるため削除対象(ignore_3D=TrueのTrueの場合)'''
    mordred_ignore3D_True_drop_column_list = [
        'BalabanJ', 'BertzCT', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA2', 
        'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 
        'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA2', 
        'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'EState_VSA1', 'EState_VSA10', 
        'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 
        'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 
        'VSA_EState9'
    ]

    '''mordredによる特徴量生成時から全ての構造式に対して全てnullを示したため削除対象(281コ)'''
    mordred_all_null_feature_list = [
        'PNSA1', 'PNSA2', 'PNSA3', 'PNSA4', 'PNSA5', 'PPSA1', 'PPSA2', 'PPSA3', 'PPSA4', 'PPSA5', 'DPSA1', 'DPSA2', 'DPSA3', 
        'DPSA4', 'DPSA5', 'FNSA1', 'FNSA2', 'FNSA3', 'FNSA4', 'FNSA5', 'FPSA1', 'FPSA2', 'FPSA3', 'FPSA4', 'FPSA5', 'WNSA1', 
        'WNSA2', 'WNSA3', 'WNSA4', 'WNSA5', 'WPSA1', 'WPSA2', 'WPSA3', 'WPSA4', 'WPSA5', 'RNCS', 'RPCS', 'TASA', 'RASA', 
        'RPSA', 'SpAbs_Dt', 'SpMax_Dt', 'SpDiam_Dt', 'SpAD_Dt', 'SpMAD_Dt', 'LogEE_Dt', 'SM1_Dt', 'VE1_Dt', 'VE2_Dt', 'VE3_Dt', 
        'VR1_Dt', 'VR2_Dt', 'VR3_Dt', 'DetourIndex', 'MAXsLi', 'MAXssBe', 'MAXssssBe', 'MAXssBH', 'MAXssssB', 'MAXsSiH3', 'MAXssSiH2', 
        'MAXsPH2', 'MAXssPH', 'MAXsssssP', 'MAXsGeH3', 'MAXssGeH2', 'MAXsssGeH', 'MAXsAsH2', 'MAXssAsH', 'MAXsssAs', 'MAXsssssAs', 
        'MAXsSeH', 'MAXdSe', 'MAXdssSe', 'MAXddssSe', 'MAXsSnH3', 'MAXsssSnH', 'MAXsPbH3', 'MAXssPbH2', 'MAXsssPbH', 'MINsLi', 
        'MINssBe', 'MINssssBe', 'MINssBH', 'MINssssB', 'MINsSiH3', 'MINssSiH2', 'MINsPH2', 'MINssPH', 'MINsssssP', 'MINsGeH3', 
        'MINssGeH2', 'MINsssGeH', 'MINsAsH2', 'MINssAsH', 'MINsssAs', 'MINsssssAs', 'MINsSeH', 'MINdSe', 'MINdssSe', 'MINddssSe', 
        'MINsSnH3', 'MINsssSnH', 'MINsPbH3', 'MINssPbH2', 'MINsssPbH', 'GeomDiameter', 'GeomRadius', 'GeomShapeIndex', 'GeomPetitjeanIndex', 
        'GRAV', 'GRAVH', 'GRAVp', 'GRAVHp', 'Lipinski', 'GhoseFilter', 'Mor01', 'Mor02', 'Mor03', 'Mor04', 'Mor05', 'Mor06', 'Mor07', 
        'Mor08', 'Mor09', 'Mor10', 'Mor11', 'Mor12', 'Mor13', 'Mor14', 'Mor15', 'Mor16', 'Mor17', 'Mor18', 'Mor19', 'Mor20', 'Mor21', 
        'Mor22', 'Mor23', 'Mor24', 'Mor25', 'Mor26', 'Mor27', 'Mor28', 'Mor29', 'Mor30', 'Mor31', 'Mor32', 'Mor01m', 'Mor02m', 'Mor03m', 
        'Mor04m', 'Mor05m', 'Mor06m', 'Mor07m', 'Mor08m', 'Mor09m', 'Mor10m', 'Mor11m', 'Mor12m', 'Mor13m', 'Mor14m', 'Mor15m', 
        'Mor16m', 'Mor17m', 'Mor18m', 'Mor19m', 'Mor20m', 'Mor21m', 'Mor22m', 'Mor23m', 'Mor24m', 'Mor25m', 'Mor26m', 'Mor27m', 
        'Mor28m', 'Mor29m', 'Mor30m', 'Mor31m', 'Mor32m', 'Mor01v', 'Mor02v', 'Mor03v', 'Mor04v', 'Mor05v', 'Mor06v', 'Mor07v', 
        'Mor08v', 'Mor09v', 'Mor10v', 'Mor11v', 'Mor12v', 'Mor13v', 'Mor14v', 'Mor15v', 'Mor16v', 'Mor17v', 'Mor18v', 'Mor19v', 
        'Mor20v', 'Mor21v', 'Mor22v', 'Mor23v', 'Mor24v', 'Mor25v', 'Mor26v', 'Mor27v', 'Mor28v', 'Mor29v', 'Mor30v', 'Mor31v', 
        'Mor32v', 'Mor01se', 'Mor02se', 'Mor03se', 'Mor04se', 'Mor05se', 'Mor06se', 'Mor07se', 'Mor08se', 'Mor09se', 'Mor10se', 
        'Mor11se', 'Mor12se', 'Mor13se', 'Mor14se', 'Mor15se', 'Mor16se', 'Mor17se', 'Mor18se', 'Mor19se', 'Mor20se', 'Mor21se', 
        'Mor22se', 'Mor23se', 'Mor24se', 'Mor25se', 'Mor26se', 'Mor27se', 'Mor28se', 'Mor29se', 'Mor30se', 'Mor31se', 'Mor32se', 
        'Mor01p', 'Mor02p', 'Mor03p', 'Mor04p', 'Mor05p', 'Mor06p', 'Mor07p', 'Mor08p', 'Mor09p', 'Mor10p', 'Mor11p', 'Mor12p', 
        'Mor13p', 'Mor14p', 'Mor15p', 'Mor16p', 'Mor17p', 'Mor18p', 'Mor19p', 'Mor20p', 'Mor21p', 'Mor22p', 'Mor23p', 'Mor24p', 
        'Mor25p', 'Mor26p', 'Mor27p', 'Mor28p', 'Mor29p', 'Mor30p', 'Mor31p', 'Mor32p', 'MOMI-X', 'MOMI-Y', 'MOMI-Z', 'PBF'
    ]

    '''mordred(ignore_3D=True)による特徴量生成時から全ての構造式に対して全てnullを示したため削除対象(281コ)'''
    mordred_ignore3D_True_all_null_feature_list = [
        'SpAbs_Dt', 'SpMax_Dt', 'SpDiam_Dt', 'SpAD_Dt', 'SpMAD_Dt', 'LogEE_Dt', 'SM1_Dt', 'VE1_Dt', 'VE2_Dt', 'VE3_Dt', 'VR1_Dt', 
        'VR2_Dt', 'VR3_Dt', 'DetourIndex', 'MAXsLi', 'MAXssBe', 'MAXssssBe', 'MAXssBH', 'MAXsssB', 'MAXssssB', 'MAXddC', 'MAXsNH3', 
        'MAXssNH2', 'MAXsssNH', 'MAXddsN', 'MAXsSiH3', 'MAXssSiH2', 'MAXsssSiH', 'MAXssssSi', 'MAXsPH2', 'MAXssPH', 'MAXsssP', 
        'MAXdsssP', 'MAXsssssP', 'MAXsGeH3', 'MAXssGeH2', 'MAXsssGeH', 'MAXssssGe', 'MAXsAsH2', 'MAXssAsH', 'MAXsssAs', 'MAXsssdAs', 
        'MAXsssssAs', 'MAXsSeH', 'MAXdSe', 'MAXssSe', 'MAXaaSe', 'MAXdssSe', 'MAXddssSe', 'MAXsSnH3', 'MAXssSnH2', 'MAXsssSnH', 
        'MAXssssSn', 'MAXsPbH3', 'MAXssPbH2', 'MAXsssPbH', 'MAXssssPb', 'MINsLi', 'MINssBe', 'MINssssBe', 'MINssBH', 'MINsssB', 
        'MINssssB', 'MINddC', 'MINsNH3', 'MINssNH2', 'MINsssNH', 'MINddsN', 'MINsSiH3', 'MINssSiH2', 'MINsssSiH', 'MINssssSi', 
        'MINsPH2', 'MINssPH', 'MINsssP', 'MINdsssP', 'MINsssssP', 'MINsGeH3', 'MINssGeH2', 'MINsssGeH', 'MINssssGe', 'MINsAsH2', 
        'MINssAsH', 'MINsssAs', 'MINsssdAs', 'MINsssssAs', 'MINsSeH', 'MINdSe', 'MINssSe', 'MINaaSe', 'MINdssSe', 'MINddssSe', 
        'MINsSnH3', 'MINssSnH2', 'MINsssSnH', 'MINssssSn', 'MINsPbH3', 'MINssPbH2', 'MINsssPbH', 'MINssssPb', 'Lipinski', 'GhoseFilter'
    ]