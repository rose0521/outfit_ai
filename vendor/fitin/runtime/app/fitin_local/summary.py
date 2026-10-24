"""Summaries and explicit demand matching; keep evidence in backend records."""
import re

PROS={
 'soft':'面料手感偏柔软，适合把触感放在前面的选择。',
 'comfortable':'穿着舒适是值得关注的优点。',
 'breathable':'透气是这款值得关注的穿着特点。',
 'non_sheer':'遮蔽性是可取之处，适合重视不透效果的需求。',
 'non_itchy':'不扎肤的穿着体验是这款的加分项。',
 'no_pilling':'已有使用体验中，起球控制表现值得关注。',
 'no_shrink':'已有洗护体验中，尺寸保持表现较好。',
 'no_fade':'已有洗护体验中，颜色保持表现较好。',
 'wash_well':'洗后表现是值得考虑的优点。',
 'well_made':'做工细节有值得肯定的地方。',
 'flattering':'版型和上身效果是这款的看点。',
 'true_size':'尺码适配有合身体验，选码时对照自身尺寸即可。',
 'warm':'保暖是值得关注的穿着特点。',
 'lightweight':'轻便是这款值得关注的穿着特点。',
}
CONS={
 'opacity':'存在偏透的反馈，介意透视效果时不宜作为首选。',
 'size_fit':'尺码适配存在分歧，选购前需要重点确认尺寸。',
 'workmanship':'做工有不满意的反馈，细节质量需要留意。',
 'fabric_comfort':'穿着感受存在分歧，舒适度不能一概而论。',
 'shape_style':'上身效果存在分歧，是否合适取决于体型和版型偏好。',
 'wash_care':'洗后表现存在负面反馈，比较在意省心打理时需要谨慎。',
 'durability':'耐用表现存在负面反馈，高频穿用时需要留意。',
 'color_accuracy':'颜色或展示一致性存在分歧，介意色差时需要留意。',
}

def summarize_reason(feature,keys,annotations):
    text=PROS.get(feature,'现有使用反馈中有可参考的正面表现。')
    if feature=='soft' and keys and all(re.search(r'\blace\b|蕾丝',annotations[k]['evidence_text'],re.I) for k in keys):
        text='蕾丝部分手感比较柔软。'
    return {'text':text,'feature':feature,'evidence_keys':keys[:2]}

def fit_assessment(candidate,intent,price_mode,annotations):
    match=candidate['match'];p=candidate['product'];criteria=[]
    if intent['categories']:criteria.append({'condition':'品类','status':'matched','basis':'product_category'})
    for state in ['supported','opposed','unknown']:
        for pref in match[state]:
            evidence=[k for k,a in annotations.items() if a['item_id']==p['item_id'] and a['feature']==pref['feature'] and a['certainty']=='reported' and (a['polarity']=='negative' if state=='opposed' else a['polarity']=='positive' or (pref['feature'] in {'warm','lightweight'} and a['polarity']=='neutral'))] if state!='unknown' else []
            criteria.append({'condition':pref['name'],'status':state,'basis':'review_analysis','evidence_keys':evidence[:2]})
    if intent['price_max'] is not None:
        criteria.append({'condition':'预算','status':'matched','basis':'demo_catalog_price' if price_mode=='demo_catalog' else 'historical_reference_price'})
    scenes={'日常':r'休闲|日常|casual|everyday','通勤':r'通勤|商务|职业|上班|\bwork\b|\boffice\b|business',
        '聚会':r'聚会|派对|\bparty\b','夏天':r'夏季|夏天|\bsummer\b','冬天':r'冬季|冬天|\bwinter\b','旅行':r'旅行|沙滩|beach|travel'}
    for scene in intent['scenario_terms']:
        hit=bool(re.search(scenes[scene],p['title']+' '+p.get('title_original',''),re.I))
        criteria.append({'condition':scene+'场景','status':'title_described' if hit else 'unknown','basis':'title_only_not_independently_verified'})
    for name in intent['unverified_constraints']:criteria.append({'condition':name,'status':'unknown','basis':'not_structurally_verified'})
    if intent['size_request']:criteria.append({'condition':'具体尺码建议','status':'unknown','basis':'size_chart_not_available'})
    unknown=list(dict.fromkeys(x['condition'] for x in criteria if x['status']=='unknown'))
    opposed=[x['name'] for x in match['opposed']]
    supported=[x['name'] for x in match['supported'] if x['name'] not in opposed]
    if opposed:
        status='partial_match' if supported else 'not_preferred'
        text='这款不作为首选：'+'、'.join(opposed)+'方面与你的要求有冲突。'
    elif unknown:
        status='partial_match' if supported else 'uncertain'
        text=('更推荐给看重'+'、'.join(supported)+'的你。' if supported else '先按款式比较，不把它列为重点推荐。')
    elif supported:
        status='matches_identified_preferences';text='更推荐给看重'+'、'.join(supported)+'的你。'
    else:status='category_match';text='品类符合，选购时重点比较下面的优缺点。'
    if intent['price_max'] is not None:text+=f" 商品库定价在{intent['price_max']:g}元预算内。"
    checks=[]
    if intent['unverified_constraints']:checks.append('购买前核对：'+'、'.join(intent['unverified_constraints'])+'。')
    if intent['size_request']:checks.append('选码需要商品尺码表和身体尺寸，先核对这两项再选。')
    return {'status':status,'summary':text,'criteria':criteria,'unconfirmed_conditions':unknown,
            'supported_preferences':supported,'opposed_preferences':opposed,'required_checks':checks,
            'all_query_conditions_verified':not unknown and not opposed}

def advice(fit,reasons,risks):
    if fit['status']=='not_preferred':return '它与当前需求有冲突，建议先看其他更匹配的款式。'
    if any('长度' in r['text'] for r in risks):return '喜欢短款可以优先看它；如果你想要更长的覆盖长度，就选下一款。'
    if any('偏透' in r['text'] for r in risks):return '能接受透感再选它；不喜欢透视效果，直接比较其他款式。'
    if risks:return '把优点和这些短板一起比较；更在意耐用、细节的话，优先选择短板更少的款式。'
    if reasons:return '可以放进优先比较名单，最后按款式和具体规格做选择。'
    return '这款先不作购买推荐，补齐关键商品资料后再筛选。'
