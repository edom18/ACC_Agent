---
trigger: always_on
---

# Python の仮想環境ルール

- 本プロジェクトでは `.venv` によって仮想環境を構築している。そのため、まずは必要な `activate` を行ったのちに各種 Python 関連のコマンドを実行すること。
- 依存関係などの解決には `uv` コマンドを用いる。`pip install` は直に実行せず、`uv pip install` を用いて依存関係を解決すること。