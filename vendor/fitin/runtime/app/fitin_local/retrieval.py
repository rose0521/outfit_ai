"""No network, no LLM: lexical retrieval, evidence-aware ranking, grounded templates."""
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

CATEGORY_NAMES={'tops':'上衣','bottoms':'下装','outerwear':'外套','all-body':'连衣裙／连体装'}
SUBTYPES=[
 ('半身裙','bottoms',r'半身裙|\bskirt\b',r'半身裙'),
 ('连衣裙','all-body',r'连衣裙|\bdress(?:es)?\b',r'连衣裙'),
 ('连体裤','all-body',r'连体裤|连体短裤|\b(?:jumpsuit|romper)\b',r'连体裤|连体短裤'),
 ('牛仔裤','bottoms',r'牛仔裤|\bjeans\b',r'牛仔裤'),
 ('打底裤','bottoms',r'打底裤|紧身裤|\bleggings\b',r'打底裤|紧身裤|瑜伽裤'),
 ('衬衫','tops',r'衬衫|\bblouse\b|\bbutton.{0,8}shirt\b',r'衬衫'),
 ('T恤','tops',r'T恤|T恤衫|T衫|\bt[ -]?shirt\b',r'[TtＴｔ]\s*[恤衫]|短袖T'),
 ('卫衣','tops',r'卫衣|\b(?:hoodie|sweatshirt)\b',r'卫衣'),
 ('针织上衣','tops',r'针织|毛衣|\bsweater\b',r'针织衫|毛衣'),
 ('开衫','outerwear',r'开衫|\bcardigan\b',r'开衫'),
 ('西装外套','outerwear',r'西装|\bblazer\b',r'西装'),
]
FEATURES=[
 ('soft','柔软',r'柔软|亲肤|软一点|软的'),
 ('breathable','透气',r'透气|不闷|怕热'),
 ('non_sheer','不透',r'不(?:要|能|想|容易|太)?透(?:明|视)?|别透|避免透|不走光|不喜欢透'),
 ('non_itchy','不扎肤',r'不扎|别扎|怕扎|扎肤|扎人'),
 ('no_pilling','不易起球',r'不起球|不容易起球|不易起球|别起球|不要起球|怕起球'),
 ('no_shrink','不易缩水',r'不缩水|不易缩水|别缩水|怕缩水'),
 ('no_fade','不易褪色',r'不褪色|不易褪色|别褪色|怕褪色'),
 ('wash_well','好打理',r'好打理|耐洗|方便洗|好洗'),
 ('well_made','做工',r'做工|缝线|别开线|不开线'),
 ('flattering','修饰身形',r'显瘦|修身|修饰身形|好版型'),
 ('true_size','尺码正常',r'尺码准|尺码正常|别偏小|不偏小'),
 ('warm','保暖',r'保暖|暖和|冬天|冬季'),
 ('lightweight','轻薄',r'轻薄|轻一点|轻便'),
]
DESCRIPTIVE={'warm','lightweight'}
FEATURE_WORDING={
 'soft':'有买家提到面料柔软，比较重视手感的话可以先看看。',
 'comfortable':'有买家觉得穿着舒服，可以作为舒适度方面的参考。',
 'breathable':'评论里有透气方面的正面反馈，偏好透气穿着感的话可以关注。',
 'non_sheer':'有买家明确说不透，值得你优先看看；具体颜色和规格仍要对照确认。',
 'non_itchy':'有买家反馈不扎肤，重视贴肤感的话可以考虑。',
 'no_pilling':'有买家提到没有起球，可以作为耐穿表现的一条参考。',
 'no_shrink':'评论里有未缩水的反馈，比较符合你对洗后表现的要求。',
 'no_fade':'有买家反馈没有褪色，可以作为选购时的参考。',
 'wash_well':'有买家觉得洗后表现不错，想找好打理的款式可以先看它。',
 'well_made':'有买家认可它的做工，可以展开看看具体评价。',
 'flattering':'有买家觉得上身比较修饰身形，喜欢这种效果的话可以考虑。',
 'true_size':'有买家觉得尺码合适，但选码时仍建议对照自己的尺寸。',
 'warm':'有买家提到穿着暖和，比较贴近你对保暖的要求。',
 'lightweight':'有买家觉得衣服轻便，偏好轻薄穿着感的话可以先看这款。',
}
RISK_WORDING={
 'opacity':'不过，有买家觉得它偏透，介意这一点的话要谨慎。',
 'size_fit':'尺码方面有不合适的反馈，选购前最好仔细核对尺寸。',
 'workmanship':'做工方面也有不满意的反馈，建议展开评论看看具体问题。',
 'fabric_comfort':'穿着感受并非全是好评，建议也看看不满意的那几条。',
 'shape_style':'也有买家不喜欢它的上身效果，是否适合自己还要看版型。',
 'wash_care':'洗后表现有负面反馈，如果很在意打理省心，建议谨慎考虑。',
 'durability':'耐用表现有负面反馈，想长期高频穿的话需要留意。',
 'color_accuracy':'有买家提到实物颜色或展示存在差异，介意色差的话要留意。',
}

def tokens(text):
    text=text.lower()
    words=re.findall(r'[a-z0-9]+',text)
    for s in re.findall(r'[\u4e00-\u9fff]+',text):
        words.extend(s[i:i+2] for i in range(len(s)-1))
        if len(s)==1:words.append(s)
    return words

def parse_query(query):
    q=query.strip()
    if not q:raise ValueError('请输入购买需求。')
    selected=[(name,cat,pat) for name,cat,pat,trigger in SUBTYPES if re.search(trigger,q,re.I)]
    category=set(x[1] for x in selected)
    if not category:
        for pattern,cat in [(r'上衣','tops'),(r'裤子|长裤|短裤|下装','bottoms'),(r'外套|大衣|夹克','outerwear'),(r'裙子','all-body')]:
            if re.search(pattern,q):category.add(cat)
    preferences=[{'feature':f,'name':name} for f,name,pat in FEATURES if re.search(pat,q)]
    max_price=None;min_price=None
    patterns=[r'(?:预算|不超过|最多|低于|小于|不高于|控制在)\s*[¥￥]?\s*(\d+(?:\.\d+)?)',r'(\d+(?:\.\d+)?)\s*(?:元|块)?\s*(?:以内|以下|之内|左右)']
    for pattern in patterns:
        m=re.search(pattern,q)
        if m:max_price=float(m.group(1));break
    interval=re.search(r'(\d+(?:\.\d+)?)\s*(?:到|至|—|-)\s*(\d+(?:\.\d+)?)\s*(?:元|块)',q)
    if interval:min_price,max_price=map(float,interval.groups())
    if max_price is not None and max_price<=0:raise ValueError('预算需要是正数。')
    if min_price is not None and min_price>max_price:raise ValueError('价格范围上下限颠倒。')
    hard=bool(re.search(r'必须|一定要|绝对不能|不能透|不要透|不接受|务必',q))
    size_request=bool(re.search(r'穿(?:什么|多大|哪个)码|选(?:什么|多大|哪个)码|尺码建议|身高|体重|\d+\s*(?:斤|公斤|厘米|cm)',q,re.I))
    unverified=[x for x in ['纯棉','百分百棉','真丝','羊绒','黑色','白色','红色','蓝色','长款','短款'] if x in q]
    out_of_scope=bool(re.search(r'男装|男士|男生|童装|儿童|婴儿|小孩',q))
    return {'query':q,'categories':sorted(category),'subtypes':[x[0] for x in selected],
        'subtype_patterns':[x[2] for x in selected],'preferences':preferences,
        'price_min':min_price,'price_max':max_price,'hard_preferences':hard,
        'size_request':size_request,'unverified_constraints':unverified,'out_of_scope':out_of_scope,
        'scenario_terms':[x for x in ['通勤','聚会','日常','夏天','冬天','旅行'] if x in q]}

class Catalog:
    def __init__(self,root,price_mode=None):
        self.root=Path(root)
        cfg=self.root/'展示与回答设置.json'
        self.presentation=json.loads(cfg.read_text(encoding='utf-8')) if cfg.exists() else {}
        self.price_mode=price_mode or self.presentation.get('price_mode','source_reference')
        if self.price_mode not in {'source_reference','demo_catalog'}:raise ValueError('未知价格使用模式。')
        # Read-only URI avoids silently creating an empty database on wrong path.
        self.db=sqlite3.connect((self.root/'fitin.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        self.products={r[0]:json.loads(r[1]) for r in self.db.execute('SELECT item_id,data_json FROM products')}
        self.assessments={r[0]:json.loads(r[1]) for r in self.db.execute('SELECT item_id,data_json FROM product_assessments')}
        self.annotations={r[0]:json.loads(r[1]) for r in self.db.execute('SELECT annotation_key,data_json FROM review_aspects')}
        self.reviews={r[0]:json.loads(r[1]) for r in self.db.execute('SELECT review_key,data_json FROM reviews')}
        self.docs={};self.df=Counter()
        for item,p in self.products.items():
            a=self.assessments[item]
            text=p['title']+' '+p.get('title_original','')+' '+CATEGORY_NAMES[p['semantic_category']]
            # Original/translated review context helps retrieve terminology, but
            # relevance is not used as evidence that a claim is positive.
            text+=' '+' '.join(r['review_text'] for r in self.reviews.values() if r['item_id']==item)
            self.docs[item]=Counter(tokens(text))
            self.df.update(self.docs[item].keys())
        self.avglen=sum(sum(v.values()) for v in self.docs.values())/max(len(self.docs),1)

    def close(self):self.db.close()

    def display_price(self,product):
        return f"¥{product['price']:.2f}" if self.price_mode=='demo_catalog' else product['price_display']

    def reason_text(self,feature,evidence_keys,position):
        from .summary import summarize_reason
        return summarize_reason(feature,evidence_keys,self.annotations)

    def get_product_assessment(self,item_id):
        if item_id not in self.products:raise ValueError('商品编号不存在。')
        return {'product':self.products[item_id],'assessment':self.assessments[item_id]}

    def get_review_evidence(self,item_id,evidence_keys=None,limit=12):
        if item_id not in self.products:raise ValueError('商品编号不存在。')
        keys=evidence_keys if evidence_keys is not None else [k for k,a in self.annotations.items() if a['item_id']==item_id]
        out=[]
        for k in dict.fromkeys(keys):
            if k not in self.annotations:raise ValueError('证据编号不存在。')
            a=self.annotations[k]
            if a['item_id']!=item_id:raise ValueError('证据不属于当前商品。')
            r=self.reviews[a['review_key']]
            out.append({**a,'review_text_zh':r['review_text'],'original_review_text':r['original_text'],
                'rating':r['rating'],'source_url':r['source_url'],'source_line':r['source_line'],
                'source_csv_row':r['source_csv_row'],'timestamp':r['timestamp']})
            if len(out)>=limit:break
        return out

    def bm25(self,item,query):
        counts=self.docs[item];length=sum(counts.values());score=0.;N=len(self.docs)
        for t in set(tokens(query)):
            f=counts[t]
            if f:
                idf=math.log(1+(N-self.df[t]+.5)/(self.df[t]+.5))
                score+=idf*f*2.2/(f+1.2*(.25+.75*length/max(self.avglen,1)))
        return score

    def search_products(self,query,limit=5,allow_estimated_budget=False):
        intent=parse_query(query);results=[];excluded=Counter();prefs=intent['preferences']
        for item,p in self.products.items():
            if intent['out_of_scope']:excluded['out_of_scope']+=1;continue
            if intent['hard_preferences'] and (intent['unverified_constraints'] or intent['size_request']):
                excluded['hard_condition_without_required_data']+=1;continue
            if intent['categories'] and p['semantic_category'] not in intent['categories']:
                excluded['category']+=1;continue
            if intent['subtype_patterns'] and not any(re.search(pat,p['title']+' '+p.get('title_original',''),re.I) for pat in intent['subtype_patterns']):
                excluded['subtype']+=1;continue
            if intent['price_max'] is not None:
                if p['price_is_estimated'] and not allow_estimated_budget and self.price_mode!='demo_catalog':
                    excluded['estimated_budget']+=1;continue
                if p['price'] is None or p['price']>intent['price_max'] or (intent['price_min'] is not None and p['price']<intent['price_min']):
                    excluded['budget']+=1;continue
            a=self.assessments[item];matched=[];opposed=[];unknown=[];evidence=[];support_count=0
            for pref in prefs:
                f=pref['feature'];v=a['features'].get(f,{})
                pos=v.get('positive',[]);neg=v.get('negative',[])
                if f in DESCRIPTIVE:pos=sorted(set(pos+v.get('neutral',[])))
                if intent['hard_preferences']:
                    # Hedged evidence (e.g. "not completely see-through") cannot
                    # satisfy an absolute user constraint.
                    allowed={x['review_key'] for x in self.annotations.values() if x['item_id']==item and x['feature']==f and x['certainty']=='reported' and not x.get('hedged') and (x['polarity']=='positive' or (f in DESCRIPTIVE and x['polarity']=='neutral'))}
                    pos=[k for k in pos if k in allowed]
                if pos:matched.append(pref)
                support_count+=min(len(pos),3)
                if neg:opposed.append(pref)
                if not pos and not neg:unknown.append(pref)
                if pos:
                    evidence.extend(k for k,x in self.annotations.items() if x['item_id']==item and x['feature']==f and x['certainty']=='reported' and (x['polarity']=='positive' or (f in DESCRIPTIVE and x['polarity']=='neutral')))
            if intent['hard_preferences'] and (opposed or unknown):
                excluded['hard_evidence_constraint']+=1;continue
            # Query relevance and preference evidence are separate sorting keys;
            # star ratings are never treated as objective quality or value scores.
            lexical=self.bm25(item,query)
            support=len(matched)-len(opposed)
            rank=(not bool(opposed),support,support_count+.2*min(a['review_count'],5),lexical,item)
            results.append({'item_id':item,'product':p,'assessment':a,
                'match':{'supported':matched,'opposed':opposed,'unknown':unknown,
                         'lexical_score':round(lexical,4),'is_quality_score':False},
                'support_evidence_keys':list(dict.fromkeys(evidence))[:5],
                '_rank':rank})
        results.sort(key=lambda x:x['_rank'],reverse=True)
        # For a category-free query with no recognized preference, don't pretend
        # lexical absence implies a meaningful product recommendation.
        if not intent['categories'] and not prefs:
            results=[x for x in results if x['match']['lexical_score']>0]
        for r in results:r.pop('_rank')
        return {'intent':intent,'candidates':results[:max(1,min(limit,20))],
                'candidate_count':len(results),'excluded':dict(excluded),
                'allow_estimated_budget':allow_estimated_budget,
                'price_mode':self.price_mode,
                'ranking_method':'满足识别的品类/预算条件后，按偏好反向反馈、偏好支持数、有限证据量及BM25相关性排序；非质量模型'}

    def compare_products(self,item_ids):
        return [self.get_product_assessment(item) for item in dict.fromkeys(item_ids)]

    def recommend(self,query,limit=3,allow_estimated_budget=False):
        from .summary import CONS,fit_assessment,advice
        result=self.search_products(query,limit,allow_estimated_budget)
        recommendations=[]
        for c in result['candidates']:
            item=c['item_id'];a=c['assessment'];reasons=[];risks=[];uncertain=[]
            for pref in c['match']['supported']:
                keys=[k for k in c['support_evidence_keys'] if self.annotations[k]['feature']==pref['feature']]
                if keys:reasons.append(self.reason_text(pref['feature'],keys,len(recommendations)))
            if not reasons:
                for strength in a['strengths'][:2]:
                    first=self.annotations[strength['evidence_keys'][0]]
                    reasons.append(self.reason_text(first['feature'],strength['evidence_keys'],len(recommendations)))
            for risk in a['risks'][:3]:
                risks.append({'text':CONS.get(risk['aspect'],f"{risk['aspect_zh']}方面存在分歧，需要进一步确认。"),'evidence_keys':risk['evidence_keys'][:2]})
            short_keys=[k for k,x in self.annotations.items() if x['item_id']==item and x['feature']=='short_length' and x['certainty']=='reported']
            if short_keys and len(risks)<3:risks.append({'text':'长度可能偏短，如果不喜欢短款，建议先确认具体长度。','evidence_keys':short_keys[:2]})
            if not reasons:uncertain.append('没有识别到足以支持具体优点的评论证据，主要因品类或关键词匹配入选。')
            if c['match']['unknown']:uncertain.append('以下偏好缺少明确反馈：'+'、'.join(x['name'] for x in c['match']['unknown'])+'。')
            if c['match']['opposed']:uncertain.append('存在与以下偏好相反的反馈：'+'、'.join(x['name'] for x in c['match']['opposed'])+'。')
            if a['review_count']<5:uncertain.append(f"仅有{a['review_count']}条采样评论，不能推断普遍表现。")
            uncertain.append('自动规则分析尚未人工复核。')
            uncertain.append('评论关联父商品，可能涉及不同颜色和尺码，不代表全部规格。')
            keys=list(dict.fromkeys(k for row in reasons+risks for k in row['evidence_keys']))
            user_notes=[]
            fit=fit_assessment(c,result['intent'],self.price_mode,self.annotations)
            recommendations.append({'item_id':item,'title':c['product']['title'],
                'price_display':self.display_price(c['product']),'price_type':c['product']['price_type'],
                'price_source_display':c['product']['price_display'],'price_mode':self.price_mode,
                'review_count':a['review_count'],'evidence_level':a['evidence_level_zh'],
                'recommendation_label':'可优先核对的候选' if reasons and not c['match']['opposed'] else '谨慎参考的候选',
                'reasons':reasons,'risks':risks,'uncertainties':uncertain,'user_notes':user_notes,
                'pros':[x['text'] for x in reasons],'cons':[x['text'] for x in risks],
                'fit_assessment':fit,'purchase_advice':advice(fit,reasons,risks),
                'evidence':self.get_review_evidence(item,keys,limit=12)})
        warnings=['本结果基于历史公开研究样本，不代表当前在售、全网最好或客观质量认证。']
        if result['intent']['price_max'] is not None and self.price_mode!='demo_catalog':
            warnings.append('预算按历史来源价或买家自述参考价筛选，不能确认当前成交价。')
            if allow_estimated_budget:warnings.append('已显式启用演示估算价筛选；预算符合仅为演示值，不可用于真实购买决策。')
            else:warnings.append('估算价商品未用于预算筛选。')
        if result['intent']['size_request']:warnings.append('当前没有完整尺码表与足够身材样本，不能据此确定应买哪个尺码。')
        if result['intent']['unverified_constraints']:warnings.append('下列材质、颜色或长度条件尚未做结构化核验，仅可作为检索文字参考，不能认定已满足：'+'、'.join(result['intent']['unverified_constraints'])+'。')
        if result['intent']['scenario_terms']:warnings.append('场景按标题和评论文字匹配，适用场景尚未独立核实。')
        if result['intent']['out_of_scope']:warnings.append('当前库为女装样本，不覆盖男装或童装。')
        if not recommendations:warnings.append('当前已识别的条件下没有足够候选；没有放宽条件或编造商品。')
        if not result['intent']['categories']:warnings.append('尚未识别明确品类；可补充“连衣裙、半身裙、衬衫、T恤、外套”等。')
        pref_names='、'.join(x['name'] for x in result['intent']['preferences'])
        if recommendations:
            opening=(f"你比较在意{pref_names}，我会先看下面这几款。" if pref_names else '我先从当前商品库里挑了几款，给你对比一下各自的优点和要注意的地方。')
        else:opening='当前商品库没有符合这些筛选条件的推荐。可以调整预算或品类后再挑。'
        user_notices=[]
        if result['intent']['size_request']:user_notices.append('选码请先对照商品尺码表和身体尺寸；身高、体重用于辅助。')
        if result['intent']['unverified_constraints']:user_notices.append('选购时重点核对：'+'、'.join(result['intent']['unverified_constraints'])+'。成分看标签，尺寸看尺码表，不只看标题。')
        if result['intent']['hard_preferences'] and result['intent']['unverified_constraints']:user_notices.append('这组商品不作为这些必选条件下的推荐，先补齐对应商品资料再筛选。')
        if result['intent']['out_of_scope']:user_notices.append('目前这份商品库只有女装，男装和童装还没有收录。')
        price_notice=('本页价格为课程演示商品库设定价，部分来自估算，不代表实时市场售价。' if self.price_mode=='demo_catalog' else '价格为历史或评论参考；标记为估算的价格不是实际售价。')
        answer={'query':query,'method':'offline_rules_bm25_grounded_templates_v1',
            'intent':result['intent'],'recommendations':recommendations,'warnings':warnings,
            'opening':opening,'user_notices':user_notices,'price_mode':self.price_mode,'price_notice':price_notice,
            'candidate_count':result['candidate_count'],'excluded':result['excluded'],
            'api_used':False,'trained_llm_used':False}
        self.check_answer(answer)
        return answer

    def check_answer(self,answer):
        for rec in answer['recommendations']:
            item=rec['item_id'];assert item in self.products
            assert rec['price_display']==self.display_price(self.products[item])
            assert rec['price_source_display']==self.products[item]['price_display']
            for claim in rec['reasons']+rec['risks']:
                assert claim['evidence_keys'],'Uncited review claim'
                if claim.get('quoted_text'):
                    assert claim['quoted_text'] in self.reviews[claim['quote_review_key']]['review_text']
                    assert self.reviews[claim['quote_review_key']]['item_id']==item
                for k in claim['evidence_keys']:
                    a=self.annotations[k]
                    assert a['item_id']==item and a['review_key'] in self.reviews
                    text=self.reviews[a['review_key']]['original_text' if a['source_language']=='original' else 'review_text']
                    assert text[a['evidence_start']:a['evidence_end']]==a['evidence_text']
            for criterion in rec['fit_assessment']['criteria']:
                if criterion['basis']=='review_analysis' and criterion['status']!='unknown':
                    assert criterion.get('evidence_keys')
                    assert all(self.annotations[k]['item_id']==item for k in criterion['evidence_keys'])
            if self.products[item]['price_is_estimated']:
                assert rec['price_type']=='category_median_estimate'
                if self.price_mode=='source_reference':assert '估算价' in rec['price_display']
                else:assert answer['price_mode']=='demo_catalog' and '估算' in answer['price_notice']
        return True

def markdown(answer):
    out=[answer.get('opening',f"你的需求：{answer['query']}"),'']
    for i,r in enumerate(answer['recommendations'],1):
        out.extend([f"{i}. {r['title']} · {r['price_display']}"])
        out.append('  适合度：'+r['fit_assessment']['summary'])
        if r['pros']:out.append('  优点：'+' '.join(r['pros']))
        if r['cons']:out.append('  需要注意：'+' '.join(r['cons']))
        out.append('  建议：'+r['purchase_advice'])
        out.extend('  '+u for u in r.get('user_notes',[]))
        out.append('')
    out.extend(answer.get('user_notices',[]))
    out.extend(['',answer.get('price_notice','')])
    return '\n'.join(out)+'\n'
