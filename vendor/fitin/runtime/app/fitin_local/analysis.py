"""Conservative, bilingual rule proposals. Outputs are not human ground truth."""
import hashlib
import re
from collections import Counter, defaultdict

ASPECTS = {
    'fabric_comfort':'面料与舒适度', 'workmanship':'做工', 'shape_style':'版型与外观',
    'size_fit':'尺码与合身度', 'opacity':'遮蔽性与透视', 'wash_care':'洗后表现',
    'durability':'起球与耐用反馈', 'color_accuracy':'颜色与描述一致性',
    'warmth_weight':'厚薄与保暖描述', 'overall_impression':'整体主观感受',
}
POLARITIES = {'positive','negative','neutral','mixed'}

# aspect, polarity, feature, expression. Specific negations are handled before
# general words. All evidence spans refer to the exact stored source text.
RULES = [
 ('fabric_comfort','positive','soft',r'\bsoft(?:ness)?\b'),
 ('fabric_comfort','positive','comfortable',r'\b(?:comfortabl[ey]|comfy|cozy)\b'),
 ('fabric_comfort','negative','comfortable',r'\buncomfortable\b'),
 ('fabric_comfort','positive','breathable',r'\b(?:breathable|breathes? well|breathes?)\b'),
 ('fabric_comfort','negative','non_itchy',r'\b(?:itchy|scratchy|scratchiness)\b'),
 ('fabric_comfort','negative','comfortable',r'\b(?:stiff|plastic[ -]?y|rough)\b'),
 ('workmanship','positive','well_made',r'\b(?:well[ -]made|well[ -]constructed|good stitching|seams? (?:are |is )?(?:well finished|sewn tight and straight))\b'),
 ('workmanship','negative','well_made',r'\b(?:cheaply made|poor(?:ly)? (?:made|stitching|sewn)|loose threads?|unfinished|messed up seam|buttons?[^.!?\n]{0,20}missing|missing buttons?)\b'),
 ('shape_style','positive','flattering',r'\bflattering\b'),
 ('shape_style','negative','flattering',r'\bunflattering\b'),
 ('shape_style','neutral','boxy',r'\bboxy\b'),
 ('size_fit','positive','true_size',r'\b(?:fits? (?:really |very |so |just |me )?(?:well|great|perfect(?:ly)?|nicely|beautifully)|fits? (?:as expected|true to size)|true to size|perfect fit|good fit)\b'),
 ('size_fit','negative','true_size',r'\b(?:too (?:small|large|big|tight)|did(?:n.t| not) fit|does(?:n.t| not) fit|could(?:n.t| not) (?:fit|zip)|ill fitting)\b'),
 ('size_fit','neutral','runs_small',r'\b(?:runs? (?:very |really |a little |quite )?small|sizes? run small)\b'),
 ('size_fit','neutral','runs_large',r'\b(?:runs? (?:very |really |a little |quite )?(?:large|big)|sizes? run (?:large|big))\b'),
 ('size_fit','neutral','short_length',r'\b(?:too short|(?:a (?:bit|little|trifle)|slightly|kind of) short|shorter than (?:expected|I (?:thought|wanted|hoped)))\b'),
 ('size_fit','neutral','long_length',r'\b(?:too long|longer than (?:expected|I (?:thought|wanted|hoped)))\b'),
 ('opacity','negative','non_sheer',r'\b(?:see[ -]?through|sheer|transparent)\b'),
 ('opacity','positive','non_sheer',r'\b(?:opaque|squat[ -]?proof)\b'),
 ('wash_care','positive','wash_well',r'\b(?:wash(?:es|ed)? (?:really |very )?(?:well|great|beautifully)|holds? up well after (?:cleaning|washing)|no shrinking)\b'),
 ('wash_care','negative','no_shrink',r'\b(?:shrink(?:s|ing)?|shrunk)\b'),
 ('wash_care','negative','no_fade',r'\b(?:fade[ds]?|fading)\b'),
 ('durability','negative','no_pilling',r'\b(?:pill(?:s|ing|y)?|pilled)\b'),
 ('durability','negative','durable',r'\b(?:snag(?:s|ged|ging)?|shed(?:s|ding)?|fray(?:s|ed|ing)?|ripped|tore|tear(?:s|ing)?|unravel(?:s|ed|ing)?|peel(?:s|ed|ing)?|fell apart|fall(?:s|ing)? apart)\b'),
 ('durability','positive','durable',r'\b(?:durable|held up (?:well|beautifully)|holds? up (?:well|beautifully))\b'),
 ('color_accuracy','positive','color_true',r'\b(?:true to (?:the )?(?:color|colour|picture)|colou?rs?[^.!?\n]{0,25}(?:as (?:shown|pictured)|exactly as shown)|exactly as pictured)\b'),
 ('color_accuracy','negative','color_true',r'\b(?:(?:wrong|incorrect) colou?r|colou?r[^.!?\n]{0,30}(?:not (?:the same|as|what)|different|darker than)|not (?:the same|as)[^.!?\n]{0,15}colou?r)\b'),
 ('warmth_weight','neutral','warm',r'\b(?:warm|toasty)\b'),
 ('warmth_weight','neutral','lightweight',r'\b(?:light[ -]?weight|light and|light material)\b'),
 ('warmth_weight','neutral','thin',r'\bthin\b'),
 ('warmth_weight','neutral','thick',r'\bthick(?:er)?\b'),
 ('overall_impression','positive','general_like',r'\b(?:love(?:d)? (?:it|this|these|the)|great quality|good quality|nice quality)\b'),
 ('overall_impression','positive','general_like',r'^\s*(?:perfect|great|nice|good|excellent|fabulous)\s*$'),
 ('overall_impression','negative','general_like',r'\b(?:poor quality|bad quality|horrible|awful|hate(?:d)? (?:it|this|these)|disappointed)\b'),
]
COMPILED=[(a,p,f,re.compile(pattern,re.I)) for a,p,f,pattern in RULES]
ZH_RULES=[
 ('fabric_comfort','positive','soft',r'柔软'),('fabric_comfort','positive','comfortable',r'舒服|舒适'),
 ('fabric_comfort','negative','comfortable',r'不舒服'),('fabric_comfort','positive','breathable',r'透气'),
 ('fabric_comfort','negative','non_itchy',r'扎肤|扎人|有点扎|很扎'),
 ('workmanship','positive','well_made',r'做工好|做工不错'),('workmanship','negative','well_made',r'开线|缝坏|做工差|做工廉价'),
 ('size_fit','positive','true_size',r'很合身|非常合身|尺码正常|正合适'),
 ('size_fit','negative','true_size',r'太小|太大|太紧|不合身'),
 ('size_fit','neutral','runs_small',r'尺码偏小'),('size_fit','neutral','runs_large',r'尺码偏大'),
 ('opacity','negative','non_sheer',r'很透|有点透|透视|透明'),
 ('opacity','positive','non_sheer',r'不透'),('wash_care','negative','no_shrink',r'缩水'),
 ('wash_care','negative','no_fade',r'褪色'),('durability','negative','no_pilling',r'起球'),
 ('durability','negative','durable',r'勾丝|掉毛|散边|破了'),
 ('overall_impression','positive','general_like',r'很喜欢|很好|不错'),
 ('overall_impression','negative','general_like',r'失望|讨厌|糟糕'),
]

def key(*parts):return hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:24]

def clauses(text):
    # Contrast is a boundary, so "soft but too tight" is not one sentiment unit.
    sep=re.compile(r'[.!?;\n。！？；]+|\b(?:but|however|although|though|except)\b|但是|不过|可是|然而',re.I)
    start=0
    for m in sep.finditer(text):
        if text[start:m.start()].strip():yield start,m.start(),text[start:m.start()]
        start=m.end()
    if text[start:].strip():yield start,len(text),text[start:]

def negate(prefix):
    s=prefix.lower().replace('’',"'")
    if re.search(r'\bnot only\s*$',s):return False
    if re.search(r"\b(?:don't|do not|didn't|did not|not)\s+(?:like|love|want|recommend|buy|purchase)\b",s):return False
    if re.search(r'\bnot (?:sure|certain)\b',s):return False
    return bool(re.search(r"\b(?:not|no|never|without|isn't|wasn't|aren't|weren't|don't|doesn't|didn't|hasn't|haven't)\b(?:\W+\w+){0,3}\W*$",s))

def extract(review):
    original=review.get('original_text','');zh=review['review_text']
    # Spanish/non-English text falls back to the reviewed Chinese translation.
    english=bool(re.search(r'\b(?:the|this|it|is|and|fit|love|very|shirt|dress|soft|cheap|small|great|nice|comfortable|too|runs|not)\b',original,re.I)) or any(p.search(original) for a,pol,f,p in COMPILED)
    text=original if english else zh
    language='original' if english else 'zh_translation'
    rules=COMPILED if english else [(a,p,f,re.compile(r)) for a,p,f,r in ZH_RULES]
    found=[]
    for start,end,clause in clauses(text):
        for aspect,polarity,feature,pattern in rules:
            for m in pattern.finditer(clause):
                p=polarity;before=clause[max(0,m.start()-55):m.start()]
                if english and negate(before) and p in {'positive','negative'}:
                    p='negative' if p=='positive' else 'positive'
                if not english and re.search(r'不(?:是|太|怎么)?$',before) and not m.group().startswith('不') and p in {'positive','negative'}:
                    p='negative' if p=='positive' else 'positive'
                # A partial negation is not proof of opacity or comfort.
                limited=bool(re.search(r'\b(?:not (?:completely|entirely|very|as)|hardly|somewhat|slightly|a (?:little|bit))\b',clause,re.I))
                if aspect=='opacity' and re.search(r'not (?:completely|entirely)',clause,re.I):p='neutral'
                # "Don't think it will last" reverses a positive durability prediction.
                if aspect=='durability' and p=='positive' and re.search(r"\b(?:don't|do not|can't|cannot)\s+(?:think|expect|see)",before,re.I):p='negative'
                speculative=bool(re.search(r"\b(?:might|may|could|probably|hope|hopefully|expect|afraid|worried|looks? like|seems? to|not sure|not certain|don't know|do not know)\b|可能|也许|担心|希望",clause,re.I))
                # Present liking is still a subjective report; certainty never means verified fact.
                record={'annotation_key':key(review.get('review_uid',review['review_key']),aspect,start+m.start(),start+m.end(),feature),
                    'review_key':review['review_key'],'item_id':review['item_id'],
                    'aspect':aspect,'aspect_zh':ASPECTS[aspect],'polarity':p,'feature':feature,
                    'evidence_text':text[start:end],'evidence_start':start,'evidence_end':end,
                    'matched_text':m.group(),'source_language':language,
                    'certainty':'speculative' if speculative else 'reported',
                    'hedged':limited,'label_origin':'rule_proposal',
                    'human_review_status':'pending','method_version':'bilingual_rules_v1',
                    'fact_status':'commenter_report_not_independently_verified'}
                assert text[record['evidence_start']:record['evidence_end']]==record['evidence_text']
                found.append(record)
    # One clause can contain several rule triggers; one evidence record per
    # (review, aspect, feature, polarity, span), never multiple buyer votes.
    unique={}
    for r in found:unique.setdefault((r['aspect'],r['feature'],r['polarity'],r['evidence_start'],r['evidence_end']),r)
    return list(unique.values())

def summarize(product,reviews,annotations):
    stats={};positive=[];negative=[];features={}
    for aspect,name in ASPECTS.items():
        rows=[a for a in annotations if a['aspect']==aspect]
        grouped=defaultdict(set)
        for a in rows:grouped[a['review_key']].add(a['polarity'])
        counts=Counter('mixed' if ('positive' in p and 'negative' in p) or 'mixed' in p else
                       'positive' if 'positive' in p else 'negative' if 'negative' in p else 'neutral'
                       for p in grouped.values())
        stats[aspect]={'name':name,'mentioned_reviews':len(grouped),
            'positive_reviews':counts['positive'],'negative_reviews':counts['negative'],
            'mixed_reviews':counts['mixed'],'neutral_reviews':counts['neutral'],
            'not_mentioned_reviews':len(reviews)-len(grouped),
            'evidence_keys':list(dict.fromkeys(a['annotation_key'] for a in rows))}
        # General liking and physical thickness are not inferred quality advantages.
        if aspect not in {'overall_impression','warmth_weight'}:
            for polarity,target in [('positive',positive),('negative',negative)]:
                evidence=[a for a in rows if a['polarity']==polarity and a['certainty']=='reported']
                if evidence:
                    target.append({'aspect':aspect,'aspect_zh':name,
                        'review_count':len({a['review_key'] for a in evidence}),
                        'evidence_keys':list(dict.fromkeys(a['annotation_key'] for a in evidence))[:4]})
    for feature in sorted({a['feature'] for a in annotations}):
        rows=[a for a in annotations if a['feature']==feature]
        features[feature]={p:sorted({a['review_key'] for a in rows if a['polarity']==p and a['certainty']=='reported'}) for p in ['positive','negative','neutral','mixed']}
        features[feature]['speculative']=sorted({a['review_key'] for a in rows if a['certainty']=='speculative'})
    n=len(reviews)
    level='very_limited' if n<3 else 'limited' if n<5 else 'preliminary'
    mean=sum(float(r['rating']) for r in reviews)/n if n else None
    return {'item_id':product['item_id'],'review_count':n,'sample_rating_mean':round(mean,3) if mean else None,
        'rating_counts':dict(Counter(str(r['rating']) for r in reviews)),
        'evidence_level':level,'evidence_level_zh':{'very_limited':'证据很少','limited':'证据有限','preliminary':'仅可初步汇总'}[level],
        'sampling_scope':'源文件顺序有限抽样，不代表平台全量或随机样本',
        'assessment_status':'automatic_pending_human_review','human_verified_review_count':0,
        'aspects':stats,'features':features,'strengths':positive,'risks':negative,
        'unknown_aspects':[a for a,s in stats.items() if not s['mentioned_reviews']],
        'no_objective_quality_score':True,'no_return_rate_estimate':True,
        'notes':['观点仅代表样本买家的表达，自动分析待复核。',
                 '未提及不等于表现好；星级未用于推断具体维度。',
                 '同一评论每个维度最多计一次；混合观点不强行改成单一好坏。']}
