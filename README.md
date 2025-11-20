# USD/JPY FX Prediction Web App

AIを使ったUSD/JPY為替予測アプリケーション。LightGBMを使用して次の1時間の為替動向を予測します。

## 機能

1. **リアルタイム予測**: 次の1時間の為替動向（上昇/下降）を予測
2. **24時間チャート**: 直近24時間のローソク足表示
3. **パフォーマンス分析**: 過去3ヶ月のバックテスト結果を可視化
   - 勝率、リターン、トレード数
   - 日付範囲フィルター
   - トレード箇所のマーカー表示

## 技術スタック

- **Backend**: Flask (Python)
- **Frontend**: HTML/CSS/JavaScript (Bootstrap, Plotly.js)
- **ML Model**: LightGBM
- **Data Source**: Yahoo Finance (yfinance)

## デプロイ方法

### GitHub

1. GitHubで新しいリポジトリを作成
2. リモートリポジトリを追加:
```bash
git remote add origin <your-github-repo-url>
git branch -M main
git push -u origin main
```

### Render

1. [Render](https://render.com)にログイン
2. 「New」→「Web Service」を選択
3. GitHubリポジトリを接続
4. 以下を設定:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --chdir web app:app`
   - **Environment**: Python 3
5. 「Create Web Service」をクリック

## ローカル実行

```bash
# 依存関係インストール
pip install -r requirements.txt

# サーバー起動
python web/app.py
```

http://127.0.0.1:5001 にアクセス

## モデル詳細

- **アルゴリズム**: LightGBM (Binary Classification)
- **特徴量**: 79個（移動平均、RSI、ボラティリティ、マルチタイムフレームリターン等）
- **勝率**: 約57%
- **バックテストリターン**: +4.16% (3ヶ月)
