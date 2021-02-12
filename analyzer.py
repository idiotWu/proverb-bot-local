"""感情推定"""
import os
import re
import gensim
import numpy as np
from janome.tokenizer import Token, Tokenizer
from sayings import sayings
from dataclasses import dataclass
from typing import List, Tuple


# 学習済みベクトルを読み込む
# ベクトルダウンロード：https://github.com/singletongue/WikiEntVec/releases
if os.path.exists('word_vectors.bin'):
    model = gensim.models.KeyedVectors.load_word2vec_format('word_vectors.bin', binary=True)
elif os.path.exists('word_vectors.txt'):
    model = gensim.models.KeyedVectors.load_word2vec_format('word_vectors.txt')
else:
    raise FileNotFoundError('学習済みベクトルが見つかりません')

tokenizer = Tokenizer()
# 品詞フィルタ
token_filter = re.compile(r'^(名詞|動詞|副詞|形容詞|感動詞),')
# 否定語リスト
negative_words = ['ない', 'ず', 'ぬ', 'ん']

# 感情リスト
emotions: List[str] = list(sayings.keys())
# 感情ベクトルリスト
emotion_vectors: List[np.ndarray] = []

# 各感情ベクトルを算出
for emotion in emotions:
    # synonyms = model.most_similar(emotion, topn=5)
    # synonym_vectors = [model.get_vector(word) for word, _ in synonyms]
    # 類義語のベクトルを取得
    synonym_vectors = [model.get_vector(word) for word in sayings[emotion]['synonyms']]
    # 自分自身を追加
    synonym_vectors.append(model.get_vector(emotion))
    # 平均向きを算出するために単位ベクトルと変換
    synonym_vectors = [vec / np.linalg.norm(vec) for vec in synonym_vectors]
    # 単位ベクトルの平均を算出し，感情ベクトルとしてリストに入れる
    emotion_vectors.append(np.average(synonym_vectors, axis=0))


@dataclass
class WordWithPolarity:
    """極性をもつ単語"""
    # 形態素解析により得られたトークン
    token: Token
    # 極性
    polarity: int = 1

    def is_independent(self) -> bool:
        """
        自立語であるかを判定

        Returns: 自立語であれば True，そうでなければ False
        """
        # return ',自立,' in self.token.part_of_speech
        return ',非自立,' not in self.token.part_of_speech

    def invert_polarity(self) -> None:
        """極性を反転"""
        self.polarity *= -1

    def get_vector(self) -> np.ndarray:
        """
        単語ベクトルを算出

        Returns: 単語ベクトル
        """
        # 単語ベクトルを算出し
        v = model.get_vector(self.token.surface)
        # 単位ベクトルと変換し，極性をかける
        return self.polarity * v / np.linalg.norm(v)


def get_doc_vec(doc):
    """
    入力文章の分散表現を算出
    文章ベクトルは単語ベクトルの単位ベクトル平均として算出

    Args:
        doc: 入力文章

    Returns: 文章ベクトル
    """
    # 「ありません」は「あり (動詞, 自立)」「ませ」「ん」と解析され，
    # 最も近い自立語「あり」の極性が反転されてしまうので，
    # 「ない」に置き換える
    doc = re.sub('ありません', 'ない', doc)
    # 内容語
    content_words: List[WordWithPolarity] = []

    for t in list(tokenizer.tokenize(doc)):
        # 品詞フィルタと一致すれば
        if token_filter.match(t.part_of_speech):
            content_words.append(WordWithPolarity(t))
        # 基本形が否定表現であれば
        elif t.base_form in negative_words:
            # 最も近い自立語の極性を反転
            for i in range(len(content_words)-1, -1, -1):
                w = content_words[i]
                # 自立語であれば
                if w.is_independent():
                    # 極性を反転
                    w.invert_polarity()
                    break

    if len(content_words) == 0:
        raise Exception('内容語がありません')

    # すべての内容語のベクトル
    vectors = []

    for w in content_words:
        try:
            # 単語ベクトルを算出しリストに入れる
            vectors.append(w.get_vector())
        except KeyError:
            # 辞書にない単語は飛ばす
            pass

    # 単語ベクトルの平均を取って文章ベクトルとする
    return np.average(vectors, axis=0)


def calc_similarities(vector: np.ndarray) -> List[Tuple[str, float]]:
    """
    指定の単語ベクトルとすべての感情との類似度を算出

    Args:
        vector:   単語ベクトル

    Returns: [(感情, 類似度), ...]
    """
    # すべての感情ベクトルとの cos 類似度を算出
    similarities = model.cosine_similarities(vector, emotion_vectors)

    # [0, 1] に正規化
    similarities = 0.5 + 0.5 * similarities

    # [(感情, 類似度) ...]
    result = list(zip(emotions, similarities))
    # 類似度の大きい順に並べ替える
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def get_emotions(doc: str) -> List[Tuple[str, float]]:
    """
    入力文章に対して感情推定を行う

    Args:
        doc: 入力文章

    Returns: [(感情, 類似度), ...]
    """
    # 文章ベクトルを算出
    doc_vec = get_doc_vec(doc)
    # 各感情との類似度を算出
    return calc_similarities(doc_vec)


# test
if __name__ == '__main__':
    from pprint import pprint
    import traceback

    while True:
        try:
            pprint(get_emotions(input('>>> ')))
        except Exception:
            traceback.print_exc()
        finally:
            print()
