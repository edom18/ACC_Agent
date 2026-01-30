from pydantic import BaseModel, Field
from typing import List, Optional

class CompressedCognitiveState(BaseModel):
    """
    ACCによって管理される有界な内部状態 (Bounded Internal State)。
    過去の全履歴の代わりに、このオブジェクトのみを次のターンに持ち越す。
    """
    
    # 1. Episodic trace: 直近のターンで何が起きたか（短期記憶）
    episodic_trace: str = Field(..., description="直近の観測事実、ユーザー入力、ツールの結果の簡潔な記録")
    
    # 2. Semantic gist: 現在の対話の要点・意味
    semantic_gist: str = Field(..., description="現在の状況やトピックの抽象的な要約")
    
    # 3. Focal entities: 重要なエンティティ（ID, サーバー名, 固有名詞など）
    focal_entities: List[str] = Field(..., description="現在注目すべき重要なエンティティのリスト（型情報含むとなお良い）")
    
    # 4. Relational map: 因果関係や依存関係
    relational_map: List[str] = Field(default=[], description="イベント間の因果関係や時間的依存関係")
    
    # 5. Goal orientation: 現在の目標（不変）
    goal_orientation: str = Field(..., description="解決すべきタスクの全体的な目標")
    
    # 6. Constraints: 守るべき制約事項（不変・最重要）
    constraints: List[str] = Field(..., description="絶対に違反してはならないルール、禁止事項、ポリシー")
    
    # 7. Predictive cue: 次に予想される展開
    predictive_cue: Optional[str] = Field(None, description="次に実行すべきステップや予想される展開")
    
    # 8. Uncertainty signal: 不確実な情報の明示
    uncertainty_signal: str = Field(..., description="まだ確認されていない事項やリスクレベル")
    
    # 9. Retrieved artifacts: 外部情報の参照（内容はコピーせず参照のみ）
    retrieved_artifacts: List[str] = Field(default=[], description="判断の根拠となった外部情報の参照IDや要点")
