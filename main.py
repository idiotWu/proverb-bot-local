"""名言ボット"""
import traceback
from analyzer import get_emotions
from sayings import pick_saying


def generate_reply(message: str) -> str:
    """
    返信を生成

    Args:
        message: 入力文章

    Returns: 返信内容
    """
    try:
        # 感情を推定
        emotions = get_emotions(message)
        print(f'[Emotions] {emotions}')

        # 最上位の感情を選ぶ
        emotion_chosen = emotions[0][0]
        # 名言を取得
        celebrity, saying = pick_saying(emotion_chosen)

        # 返信内容を生成
        reply = f'{emotion_chosen}を感じていますか？{celebrity}はこう言いました、「{saying}」'
        reply += '\n\n感情推定結果：'
        # 上位 3 つの感情を表示
        reply += '、'.join(f'{emo} ({prob:.3f})' for emo, prob in emotions[:3])
    except Exception:
        traceback.print_exc()
        reply = 'あなたの感情を推定できませんでした。'

    return reply


if __name__ == '__main__':
    print('私は何度でも蘇る🤪')
    while True:
        try:
            print(generate_reply(input('>>> ')))
        except Exception:
            traceback.print_exc()
        finally:
            print()
