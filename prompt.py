SYSTEM = ("You turn a spoken line into a compact JSON scene plan. Output ONLY JSON with schema "
 '{"objects":[{"name":str (singular noun),"count":int,"relation":{"type":one of on|under|behind|in_front_of|next_to|inside|left_of|right_of,"target":str} or null}],'
 '"treatment": one of object|comparison|chart|process|hold}. Include the relation target as its own object with count 1. '
 'For abstract statements use an empty objects list and pick the treatment; use "hold" when nothing new should be shown '
 "(fillers, pauses, repetition, restating).")
def messages(text): return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
