
# start_today_intern  
README

Matrix Factrizationを用いて、ml-100kのデータセットから、映画のレコメンドを表示させる。  
latent factorとしてはUesr IDとItem IDの2つに分解している。  

使用方法としては以下のようになる。但し、User IDはリストで入力すること。　　

import MF_intern  
from MF_intern import main  　
m = main.MatrixFactorization()    
m.recommend([1,20])  
