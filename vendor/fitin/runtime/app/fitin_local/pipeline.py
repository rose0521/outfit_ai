"""Build a self-contained offline SQLite/evidence/training-preparation bundle."""
import collections
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
from contextlib import closing
from urllib.parse import urlsplit

from . import VERSION
from .analysis import ASPECTS, POLARITIES, RULES, ZH_RULES, extract, key, summarize

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def jsonlines(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def write_lines(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def split_groups(products,reviews):
    parent={p['item_id']:p['item_id'] for p in products}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    originals={};translated={}
    def norm(s):return re.sub(r'[^a-z0-9\u4e00-\u9fff]','',s.lower())
    # Exact duplicate English OR Chinese reviews cannot straddle train/test.
    # Short generic comments remain grouped as well; they are excluded from training exports.
    for r in reviews:
        for text,seen in [(r['original_text'],originals),(r['review_text'],translated)]:
            n=norm(text)
            if n:
                if n in seen:union(r['item_id'],seen[n])
                else:seen[n]=r['item_id']
    # High lexical overlap near duplicates, for meaningful-length originals.
    prepared=[]
    for r in reviews:
        n=norm(r['original_text'])
        if len(n)>=40:prepared.append((r['item_id'],len(n),{n[i:i+5] for i in range(len(n)-4)}))
    near=0
    for i,(a,la,sa) in enumerate(prepared):
        for b,lb,sb in prepared[:i]:
            if a==b or min(la,lb)/max(la,lb)<.9:continue
            if len(sa&sb)/max(1,len(sa|sb))>=.92:union(a,b);near+=1
    groups=collections.defaultdict(list)
    for item in parent:groups[find(item)].append(item)
    # Deterministic allocation of whole connected groups, largest first.
    split_names=['train','validation','test'];targets={'train':.7*len(parent),'validation':.15*len(parent),'test':.15*len(parent)}
    totals=collections.Counter();mapping={}
    for root,items in sorted(groups.items(),key=lambda x:(-len(x[1]),key('split-v1',x[0]))):
        choice=min(split_names,key=lambda s:(totals[s]/max(targets[s],1),s))
        for item in items:mapping[item]={'group_id':'group_'+key(root),'split':choice}
        totals[choice]+=len(items)
    report={'方法':'商品及原文/译文重复连接分组后整体划分；近重复为归一化5字符片段Jaccard>=0.92且长度比>=0.9',
        '目标比例':'70/15/15，按整组划分不保证精确比例','商品分布':dict(totals),
        '分组数':len(groups),'最大组商品数':max(map(len,groups.values())),'近重复连接数':near,
        '随机规则':'固定SHA256排序，无运行随机性','分组版本':'split-v1'}
    return mapping,report

def validate_submission(records,reviews):
    byid={r['review_key']:r for r in reviews};seen=set();approved=[]
    for r in records:
        if r.get('status')!='approved':continue
        rk=r.get('review_key')
        if rk not in byid or rk in seen:raise ValueError('标注评论编号未知或重复：'+str(rk))
        seen.add(rk);source=byid[rk]
        if r.get('source_sha256')!=source['source_sha256']:raise ValueError('标注原文摘要不匹配：'+rk)
        if not str(r.get('reviewed_by','')).strip():raise ValueError('通过标注需要填写复核者代号：'+rk)
        if not isinstance(r.get('labels'),list):raise ValueError('labels必须为列表：'+rk)
        output=[]
        for label in r['labels']:
            if label.get('aspect') not in ASPECTS or label.get('polarity') not in POLARITIES:raise ValueError('维度或极性无效：'+rk)
            language=label.get('source_language','original')
            if language not in {'original','zh_translation'}:raise ValueError('证据语言无效：'+rk)
            text=source['original_text'] if language=='original' else source['review_text']
            start,end=label.get('evidence_start'),label.get('evidence_end')
            if type(start) is not int or type(end) is not int or not 0<=start<end<=len(text):raise ValueError('证据位置无效：'+rk)
            if text[start:end]!=label.get('evidence_text'):raise ValueError('证据片段与原文不符：'+rk)
            if label.get('certainty','reported') not in {'reported','speculative'}:raise ValueError('证据性质无效：'+rk)
            if label.get('feature','unspecified') not in {x[2] for x in RULES+ZH_RULES}|{'unspecified'}:raise ValueError('特征代码无效：'+rk)
            output.append({**label,'annotation_key':key(source['review_uid'],label['aspect'],start,end,label.get('feature','unspecified'),label['polarity'],'reviewed'),
                'review_key':rk,'item_id':source['item_id'],'aspect_zh':ASPECTS[label['aspect']],
                'label_origin':'user_reviewed','human_review_status':'approved',
                'certainty':label.get('certainty','reported'),'feature':label.get('feature','unspecified'),
                'method_version':'user_annotation_v1','fact_status':'commenter_report_not_independently_verified'})
        approved.append({**r,'labels':list({x['annotation_key']:x for x in output}.values())})
    return approved

def training_exports(out,reviews,annotations,mapping):
    db=sqlite3.connect(out/'fitin.sqlite')
    submitted=[json.loads(x[0]) for x in db.execute("SELECT data_json FROM training_annotations WHERE status='approved'")]
    db.close()
    byreview=collections.defaultdict(list)
    for a in annotations:byreview[a['review_key']].append(a)
    template=[]
    for r in reviews:
        template.append({'review_key':r['review_key'],'item_id':r['item_id'],'review_uid':r['review_uid'],
            'source_sha256':r['source_sha256'],'original_text':r['original_text'],'review_text_zh':r['review_text'],
            'rating_for_reference_only':r['rating'],'split':mapping[r['item_id']]['split'],
            'group_id':mapping[r['item_id']]['group_id'],'status':'pending','reviewed_by':'',
            'labels':byreview[r['review_key']],'label_origin':'automatic_proposals_not_gold'})
    # Round-robin rating band/category selection; max3 per item in pilot set.
    buckets=collections.defaultdict(list);counts=collections.Counter()
    byproduct={r['item_id']:r.get('semantic_category','') for r in reviews}
    for r in sorted(template,key=lambda x:(-len({a['aspect'] for a in x['labels']}),key(x['review_key']))):
        band='low' if float(r['rating_for_reference_only'])<=2 else 'mid' if float(r['rating_for_reference_only'])==3 else 'high'
        buckets[(byproduct[r['item_id']],band)].append(r)
    pilot=[]
    while len(pilot)<240 and any(buckets.values()):
        for bucket in sorted(buckets):
            while buckets[bucket]:
                r=buckets[bucket].pop(0)
                if counts[r['item_id']]<3:
                    pilot.append(r);counts[r['item_id']]+=1;break
            if len(pilot)>=240:break
    write_lines(out/'annotations/全部待复核标注.jsonl',template)
    write_lines(out/'annotations/首批240条标注.jsonl',pilot)
    sources={r['review_key']:r for r in reviews}
    for split in ['train','validation','test']:
        gold=[];silver=[]
        for r in submitted:
            src=sources[r['review_key']]
            if mapping[src['item_id']]['split']==split:
                gold.append({'review_key':src['review_key'],'item_id':src['item_id'],
                    'group_id':mapping[src['item_id']]['group_id'],'input':{'original_text':src['original_text'],'review_text_zh':src['review_text']},
                    'target':r['labels'],'label_origin':'user_reviewed','reviewed_by':r['reviewed_by']})
        for r in template:
            if r['split']==split and len(re.sub(r'\W','',r['original_text']))>=12 and r['labels']:
                silver.append({'review_key':r['review_key'],'item_id':r['item_id'],'group_id':r['group_id'],
                    'input':{'original_text':r['original_text'],'review_text_zh':r['review_text_zh']},'target':r['labels'],
                    'label_origin':'rule_derived_silver','not_human_gold':True,
                    'usage':'仅调试格式与弱监督实验，不能作为独立准确率评测真值'})
        write_lines(out/f'training/gold_{split}.jsonl',gold)
        write_lines(out/f'training/silver_{split}.jsonl',silver)
    save(out/'training/训练准备状态.json',{'已复核评论数':len(submitted),'首批标注数':len(pilot),
        '正式训练是否已执行':False,'说明':'当前gold可能为空，未将自动标签冒充人工真值；silver仅是规则派生弱标签。',
        '训练建议':'先复核标注、冻结商品分组测试集，再训练评论维度模型；智能体回答微调需另行收集经过复核的问答。',
        '划分':'同一商品、原文/中文重复及高相似原文连接组不能跨train/validation/test。'})
    return pilot

SCENARIOS=[
 ('dress_soft','想买日常连衣裙，希望柔软一点'),
 ('shirt_care','想买衬衫，比较在意做工和好打理'),
 ('pants_pill','推荐打底裤，不容易起球，穿着舒服'),
 ('coat_warm','想买冬天的外套，保暖一点'),
 ('dress_budget','预算200元以内，想买连衣裙'),
 ('no_sheer','我想买连衣裙，必须不透'),
 ('size_unknown','我身高165厘米，体重55公斤，连衣裙该选什么码'),
 ('no_budget_match','预算1元以内想买外套'),
 ('skirt','我想买半身裙'),
 ('tee','想买T恤，希望透气'),
]

def build(source,out,annotations_path=None,price_mode='source_reference'):
    source=Path(source).resolve();out=Path(out).resolve()
    if out==source or source.is_relative_to(out):raise ValueError('输出目录不能覆盖输入目录。')
    if (out/'fitin.sqlite').exists():
        db=sqlite3.connect(out/'fitin.sqlite')
        existing_keys={x[0] for x in db.execute("SELECT review_key FROM training_annotations WHERE status='approved'")};db.close()
        incoming={x['review_key'] for x in jsonlines(annotations_path) if x.get('status')=='approved'} if annotations_path else set()
        if not existing_keys<=incoming:raise ValueError('已有已复核标注，请通过--annotations传入全部已复核记录，避免重建时丢失。')
    fields={'metadata':['item_id','semantic_category','title','price','image_path'],
            'reviews':['item_id','rating','review_text']}
    csvrows={}
    for n,f in fields.items():
        rows=readcsv(source/(n+'.csv'))
        if not rows or list(rows[0])!=f:raise ValueError('输入表头不符：'+n)
        csvrows[n]=rows
    prices={r['item_id']:r for r in jsonlines(source/'前端价格显示.jsonl')}
    originals=jsonlines(source/'评论中外文对照.jsonl');proof=jsonlines(source/'review_provenance.jsonl')
    titles={r['item_id']:r for r in jsonlines(source/'标题中英对照.jsonl')}
    products=[];reviews=[]
    for r in csvrows['metadata']:
        if urlsplit(r['image_path']).scheme not in {'http','https'}:raise ValueError('主图链接必须是HTTP(S)，不能执行本地脚本。')
        if r['item_id'] not in prices:raise ValueError('缺价格来源标记：'+r['item_id'])
        pr=prices[r['item_id']]
        if float(r['price'])!=pr['price']:raise ValueError('CSV和价格明细金额不同。')
        products.append({**r,'price':float(r['price']),'title_original':titles[r['item_id']]['原文'],
            'category_zh':{'tops':'上衣','bottoms':'下装','outerwear':'外套','all-body':'连衣裙／连体装'}[r['semantic_category']],
            'price_type':pr['price_type'],'price_is_estimated':pr['price_is_estimated'],
            'price_display':pr['price_display'],'price_note':pr['price_note'],
            'image_not_downloaded_by_this_program':True})
    ids={p['item_id'] for p in products}
    if len(ids)!=len(products):raise ValueError('商品编号重复。')
    if not len(originals)==len(proof)==len(csvrows['reviews']):raise ValueError('评论证据数量不符。')
    categories={p['item_id']:p['semantic_category'] for p in products}
    for i,(r,o,e) in enumerate(zip(csvrows['reviews'],originals,proof),2):
        if r['item_id'] not in ids or not r['item_id']==o['item_id']==e['item_id'] or o['reviews_csv_row']!=i:raise ValueError('评论关联不符。')
        if r['review_text']!=o['中文']:raise ValueError('评论中外文对照不符。')
        if not 1<=float(r['rating'])<=5:raise ValueError('评分超出1—5。')
        original=o['原文'];source_sha=hashlib.sha256(original.encode()).hexdigest()
        reviews.append({'review_key':o['key'],'review_uid':'review_'+key(r['item_id'],source_sha,e['source_line']),
            'item_id':r['item_id'],'semantic_category':categories[r['item_id']],
            'rating':float(r['rating']),'review_text':r['review_text'],'original_text':original,
            'source_sha256':source_sha,'source_url':e['source_file'],'source_line':e['source_line'],
            'source_csv_row':i,'timestamp':e.get('timestamp'),
            'source_variant_asin':e.get('source_variant_asin'),
            'translation_notes':o['核对说明'],'sample_order':'source_file_order_max20',
            'independent_buyer_identity_verified':False})
    if len({r['review_key'] for r in reviews})!=len(reviews):raise ValueError('评论编号重复。')
    mapping,split_report=split_groups(products,reviews)
    annotations=[a for r in reviews for a in extract(r)]
    approved=validate_submission(jsonlines(annotations_path),reviews) if annotations_path else []
    if approved:
        replace={r['review_key']:r['labels'] for r in approved}
        annotations=[a for a in annotations if a['review_key'] not in replace]+[a for labels in replace.values() for a in labels]
    byproduct=collections.defaultdict(list);byannotation=collections.defaultdict(list)
    for r in reviews:byproduct[r['item_id']].append(r)
    for a in annotations:byannotation[a['item_id']].append(a)
    assessments=[summarize(p,byproduct[p['item_id']],byannotation[p['item_id']]) for p in products]
    approved_keys={r['review_key'] for r in approved}
    for a in assessments:
        a['human_verified_review_count']=sum(r['review_key'] in approved_keys for r in byproduct[a['item_id']])
        if a['human_verified_review_count']:a['assessment_status']='partially_user_reviewed'
    out.mkdir(parents=True,exist_ok=True)
    if price_mode not in {'source_reference','demo_catalog'}:raise ValueError('未知价格模式。')
    save(out/'展示与回答设置.json',{'price_mode':price_mode,'answer_style':'direct_advice_v4',
        'global_price_notice':'本页价格为课程演示商品库设定价，部分来自估算，不代表实时市场售价。' if price_mode=='demo_catalog' else '价格为历史或评论参考，估算值不代表实际售价。',
        'provenance_preserved':True,'notes':'演示模式将现有数值作为项目设定价用于展示、预算与排序；不改来源，不认定为真实成交价格。'})
    for folder in ['data','annotations','training','evaluation','examples','source_contract']:(out/folder).mkdir(exist_ok=True)
    for n in fields:shutil.copy2(source/(n+'.csv'),out/'source_contract'/(n+'.csv'))
    for name in ['类目中文名称.json','汇率依据.json','价格补全明细.json','需核对的原文记录.json','原始来源说明.json',
                 '标题中英对照.jsonl','评论中外文对照.jsonl','review_provenance.jsonl','前端价格显示.jsonl']:
        shutil.copy2(source/name,out/'source_contract'/name)
    for name,rows in [('products',products),('reviews',reviews),('review_aspects',annotations),('product_assessments',assessments)]:write_lines(out/f'data/{name}.jsonl',rows)
    evidence=[{'evidence_key':a['annotation_key'],'item_id':a['item_id'],'review_key':a['review_key'],
               'aspect':a['aspect'],'polarity':a['polarity'],'source_language':a['source_language'],
               'evidence_start':a['evidence_start'],'evidence_end':a['evidence_end']} for a in annotations]
    write_lines(out/'data/assessment_evidence.jsonl',evidence)
    save(out/'training/商品分组划分.json',mapping);save(out/'training/划分说明.json',split_report)
    with closing(sqlite3.connect(out/'fitin.build.sqlite')) as db:
        db.execute('PRAGMA foreign_keys=ON')
        # Only the generated staging database is replaced after a successful build.
        for t in ['assessment_evidence','review_aspects','training_annotations','product_assessments','reviews','products','evaluation_cases']:
            db.execute('DROP TABLE IF EXISTS '+t)
        db.executescript('''
CREATE TABLE products(item_id TEXT PRIMARY KEY, semantic_category TEXT NOT NULL, data_json TEXT NOT NULL);
CREATE TABLE reviews(review_key TEXT PRIMARY KEY,item_id TEXT NOT NULL REFERENCES products(item_id),data_json TEXT NOT NULL);
CREATE TABLE review_aspects(annotation_key TEXT PRIMARY KEY,review_key TEXT NOT NULL REFERENCES reviews(review_key),item_id TEXT NOT NULL REFERENCES products(item_id),aspect TEXT NOT NULL,polarity TEXT NOT NULL,data_json TEXT NOT NULL);
CREATE TABLE product_assessments(item_id TEXT PRIMARY KEY REFERENCES products(item_id),data_json TEXT NOT NULL);
CREATE TABLE assessment_evidence(evidence_key TEXT PRIMARY KEY REFERENCES review_aspects(annotation_key),item_id TEXT NOT NULL REFERENCES products(item_id),review_key TEXT NOT NULL REFERENCES reviews(review_key),data_json TEXT NOT NULL);
CREATE TABLE training_annotations(review_key TEXT PRIMARY KEY REFERENCES reviews(review_key),status TEXT NOT NULL,data_json TEXT NOT NULL);
CREATE TABLE evaluation_cases(case_id TEXT PRIMARY KEY,data_json TEXT NOT NULL);
''')
        enc=lambda x:json.dumps(x,ensure_ascii=False)
        db.executemany('INSERT INTO products VALUES(?,?,?)',[(p['item_id'],p['semantic_category'],enc(p)) for p in products])
        db.executemany('INSERT INTO reviews VALUES(?,?,?)',[(r['review_key'],r['item_id'],enc(r)) for r in reviews])
        db.executemany('INSERT INTO review_aspects VALUES(?,?,?,?,?,?)',[(a['annotation_key'],a['review_key'],a['item_id'],a['aspect'],a['polarity'],enc(a)) for a in annotations])
        db.executemany('INSERT INTO product_assessments VALUES(?,?)',[(a['item_id'],enc(a)) for a in assessments])
        db.executemany('INSERT INTO assessment_evidence VALUES(?,?,?,?)',[(e['evidence_key'],e['item_id'],e['review_key'],enc(e)) for e in evidence])
        db.executemany('INSERT INTO training_annotations VALUES(?,?,?)',[(r['review_key'],'approved',enc(r)) for r in approved])
        db.executemany('INSERT INTO evaluation_cases VALUES(?,?)',[(cid,enc({'query':q,'human_judgment':None,'status':'pending'})) for cid,q in SCENARIOS])
        assert db.execute('PRAGMA foreign_key_check').fetchall()==[];db.commit()
    (out/'fitin.build.sqlite').replace(out/'fitin.sqlite')
    pilot=training_exports(out,reviews,annotations,mapping)
    from .retrieval import Catalog,markdown
    catalog=Catalog(out);answers=[]
    for cid,q in SCENARIOS:
        answer=catalog.recommend(q);answers.append({'case_id':cid,**answer,'human_relevance_judgment':None})
        (out/f'examples/{cid}.md').write_text(markdown(answer),encoding='utf-8')
    # Offline dialogue candidates retain tool results; not exported as SFT gold.
    dialogue_candidates=[]
    for cid,q in SCENARIOS:
        search=catalog.search_products(q,3)
        final=next(a for a in answers if a['case_id']==cid)
        tool_call_id='search_'+cid
        messages=[{'role':'user','content':q},{'role':'assistant','tool_calls':[{'id':tool_call_id,'type':'function','function':{'name':'search_products','arguments':json.dumps({'query':q,'limit':3},ensure_ascii=False)}}]},
                  {'role':'tool','tool_call_id':tool_call_id,'name':'search_products','content':json.dumps(search,ensure_ascii=False)}]
        for j,rec in enumerate(final['recommendations']):
            call_id=f'evidence_{cid}_{j}'
            messages.extend([{'role':'assistant','tool_calls':[{'id':call_id,'type':'function','function':{'name':'get_review_evidence','arguments':json.dumps({'item_id':rec['item_id'],'evidence_keys':[e['annotation_key'] for e in rec['evidence']]},ensure_ascii=False)}}]},
                {'role':'tool','tool_call_id':call_id,'name':'get_review_evidence','content':json.dumps(rec['evidence'],ensure_ascii=False)}])
        messages.append({'role':'assistant','content':markdown(final)})
        used_ids=[r['item_id'] for r in final['recommendations']]
        splits={mapping[i]['split'] for i in used_ids}
        dialogue_candidates.append({'case_id':cid,'messages':messages,'item_ids':used_ids,
            'split':next(iter(splits)) if len(splits)==1 else 'unassigned_or_cross_split_do_not_train',
            'label_origin':'deterministic_program_candidate_not_gold','human_review_status':'pending',
            'usage':'仅为有工具证据的问答草稿，复核并重新检查商品分组后才能用于微调；不得直接当正确答案。'})
    write_lines(out/'training/待复核问答候选.jsonl',dialogue_candidates)
    catalog.close();write_lines(out/'evaluation/场景运行结果.jsonl',answers)
    evaluation_queries=[q for _,q in SCENARIOS]
    for cat in ['上衣','连衣裙','外套','裤子']:
        for pref in ['柔软','透气','不透','不扎肤','不起球','不缩水','不褪色','好打理','做工好','尺码正常']:
            evaluation_queries.append(f'我想买{cat}，希望{pref}')
        for budget in [100,150,200,300,500]:evaluation_queries.append(f'预算{budget}元以内，想买{cat}')
    for cat in ['上衣','连衣裙','外套','裤子','半身裙','衬衫']:
        for condition in ['必须不透','想要纯棉款','想要黑色款','不知道选哪个尺码','比较在意使用风险']:
            evaluation_queries.append(f'推荐{cat}，{condition}')
    evaluation_queries=list(dict.fromkeys(evaluation_queries))
    while len(evaluation_queries)<100:evaluation_queries.append(f'帮我比较女装候选，优先考虑评论证据，测试场景{len(evaluation_queries)+1}')
    evaluation=[{'case_id':f'eval_{i:03d}','query':q,'status':'pending_human_evaluation',
        'relevance_labels':None,'constraint_violation':None,'unsupported_claim':None,
        'citation_correctness':None,'notes':'待标注；不预填推荐质量结论。'} for i,q in enumerate(evaluation_queries[:100],1)]
    write_lines(out/'evaluation/100条待人工评测问题.jsonl',evaluation)
    save(out/'后端工具合同.json',{'version':VERSION,'network_calls':False,'tools':[
        {'name':'search_products','arguments':{'query':'string','limit':'integer 1—20','allow_estimated_budget':'boolean, default false'},'returns':'intent/candidates/candidate_count/excluded'},
        {'name':'get_product_assessment','arguments':{'item_id':'string'},'returns':'product/assessment'},
        {'name':'get_review_evidence','arguments':{'item_id':'string','evidence_keys':'string[] optional','limit':'integer'},'returns':'same-product evidence[]'},
        {'name':'compare_products','arguments':{'item_ids':'string[]'},'returns':'product/assessment[]'},
        {'name':'recommend','arguments':{'query':'string','limit':'integer, default 3','allow_estimated_budget':'boolean, default false'},'returns':'intent/recommendations/warnings/method/api_used/trained_llm_used'}],
        'integration_note':'这些是本地Python方法，尚未部署HTTP接口。后端可封装，但应保持证据和价型字段。',
        'recommendation_display_fields':['title','price_display','fit_assessment.summary','pros','cons','purchase_advice','user_notes'],
        'audit_only_fields':['review_count','evidence','reasons.evidence_keys','risks.evidence_keys','price_type','price_source_display','uncertainties'],
        'display_rule':'智能体回答只总结优缺点、需求匹配和购买建议；不默认展示评论原话、编号、条数。后台保留证据，未知条件不能表述为已满足。'})
    stats={'版本':VERSION,'商品数':len(products),'评论数':len(reviews),'自动或复核维度记录数':len(annotations),
        '有维度记录的评论数':len({a['review_key'] for a in annotations}),
        '待人工复核评论数':len(reviews)-len(approved),'已提交复核评论数':len(approved),
        '首批标注数量':len(pilot),'价型分布':dict(collections.Counter(p['price_type'] for p in products)),
        '样本评论数量分布':dict(collections.Counter(len(byproduct[p['item_id']]) for p in products)),
        '大模型训练已执行':False,'API已接入':False,'推荐方法':'离线规则提案＋BM25检索＋证据排序＋固定回答模板',
        '价格使用模式':price_mode,'回答风格':'优缺点总结、需求匹配判断与购买建议；证据后台保留',
        '分组情况':split_report,'来源文件摘要':{n:digest(source/n) for n in ['metadata.csv','reviews.csv','评论中外文对照.jsonl','前端价格显示.jsonl']},
        '生成时间':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')}
    save(out/'运行报告.json',stats)
    from .ui import write_ui
    write_ui(out,products,reviews,assessments,annotations,pilot,answers,stats)
    write_docs(out,stats)
    return stats

def import_annotations(out,path):
    out=Path(out);reviews=jsonlines(out/'data/reviews.jsonl')
    records=validate_submission(jsonlines(path),reviews)
    db=sqlite3.connect(out/'fitin.sqlite')
    old={rk:json.loads(data) for rk,data in db.execute("SELECT review_key,data_json FROM training_annotations WHERE status='approved'")}
    for r in records:
        if r['review_key'] in old and old[r['review_key']].get('reviewed_by')!=r.get('reviewed_by') and old[r['review_key']]['labels']!=r['labels']:
            db.close();raise ValueError('同条评论已有其他复核者的不同结果，请先人工合并分歧，不能覆盖。')
    with db:
        for r in records:db.execute('INSERT OR REPLACE INTO training_annotations VALUES(?,?,?)',(r['review_key'],'approved',json.dumps(r,ensure_ascii=False)))
    db.close()
    history={'source_file_sha256':digest(path),'records':records,'previous_records':[old[r['review_key']] for r in records if r['review_key'] in old]}
    with (out/'annotations/导入历史.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(history,ensure_ascii=False)+'\n')
    # Existing generated evidence remains unchanged until an explicit rebuild.
    training_exports(out,reviews,jsonlines(out/'data/review_aspects.jsonl'),load(out/'training/商品分组划分.json'))
    return {'导入已复核评论数':len(records),'训练集已导出':True,'评价档案已自动改变':False,
            '下一步':'如需将复核标签用于评价，请build时传入--annotations；已有评价不会被静默覆盖。'}

def write_docs(out,stats):
    doc=f'''# FITIN 离线商品评价与推荐原型

已处理{stats['商品数']}件商品、{stats['评论数']}条评论，生成{stats['自动或复核维度记录数']}条维度证据记录。这个程序完全离线，不访问外部API，不下载模型，不运行云服务。

## 打开和运行

双击“打开离线查看器.cmd”，可查看商品评价、证据和预置推荐案例。双击“打开标注工作台.cmd”，可对首批240条自动提案进行复核、修改并导出JSONL。两个网页都不联网，图片显示为链接，点击图片才由用户自行访问外部网站。

双击“启动推荐问答.cmd”，输入中文需求即可运行动态检索与推荐。电脑需要Python 3.10或以上；程序仅用标准库，不用安装第三方包。当前Codex电脑也可使用已安装的内置Python。

命令行（在本包目录运行）：

```text
python app/fitin_offline.py chat --data .
python app/fitin_offline.py query --data . --query "想买柔软的连衣裙" --output 我的推荐结果.json
python app/fitin_offline.py assess --data . --item-id amazon23_B01C7UW6QQ
python app/fitin_offline.py compare --data . --item-ids amazon23_B01C7UW6QQ amazon23_B00A6PJ3AM
python app/fitin_offline.py import-annotations --data . --file 我的复核标注.jsonl
python app/fitin_offline.py build --source source_contract --out ../FITIN_复核后版本 --annotations 我的复核标注.jsonl
python -m unittest discover -s tests -v
```

## 当前实现了什么

1. 固定商品合同导入：编号、评分、中外文对照、价格类型逐行核对。
2. 双语规则预标注：面料舒适度、做工、版型、尺码、透视、洗后、耐用、色差、厚薄和整体感受。优先分析可回查的外文原文，非英语评论回退到中文译文。每条证据有原文起止位置。自动提案都标pending，不能当作人工金标。
3. 商品评价：按每条评论每个维度去重计数，保留混合、推测、未提及和证据不足。没有生成真实质量分、退货概率或实际退货率。星级均值只是样本描述，不决定维度情感或购买排序。
4. SQLite数据库与JSONL：商品、评论、维度分析、评价档案、证据、标注和待人工评测场景。
5. 本地推荐：识别部分中文品类、偏好、预算与尺码问题；BM25词法检索、偏好证据排序、固定模板解释。每项理由都经过商品编号和证据片段检查。不是自由对话大模型，不理解所有自然语言约束。
6. 训练准备：商品及重复/近重复内容按连接组分割，输出240条标注工作批次、全部待复核数据和训练/验证/测试划分。gold只有导入经用户复核的数据才会出现；silver是规则派生标签，只供调试或弱监督实验。另有10条含真实工具返回证据的问答草稿，均待复核，跨训练/测试商品组的草稿不得直接用于训练。
7. 评测准备：100条待人工评测问题及10条实际场景运行结果，评分字段留空，未伪造推荐准确率。

## 使用限制

本批是Amazon历史研究样本，不是国内电商实时商品。351件商品只有一条样本评论。买家表达不等于已独立证实的商品事实，少量样本也不代表总体。部分源评论含歧义，译文和来源说明已保留。

469件价格为同品类中位数估算，30件为历史元数据价，1件为买家自述购买参考价。界面和回答保留价格类型。普通预算查询不拿估算价当预算依据，历史/自述价格也不能承诺当前成交价。仅在显式启用--allow-estimated-budget时用于演示，并附警示。

厚薄、保暖、尺码偏大偏小是描述或适配问题，不天然意味着质量好坏。程序可提出自动观察，不能作出“全网最好”“必买”“一定不缩水”“该买L”等保证。用户提出必须不透等强约束时，未找到对应明确反馈的商品不会冒充符合。

没有人工真值时，不报告推荐准确率、召回率、NDCG或语言模型训练成绩。场景运行报告只是技术一致性检查，不是购买推荐质量的独立评测。

## 标注怎么做

在标注工作台查看原文、中文和规则提案，修改JSON标签中的aspect、polarity、feature、certainty及证据区间。原文有歧义或证据不明确时先标“需复核”，不强行通过。填写复核者代号后逐条点击通过。labels=[]表示核对后未发现本项目所列维度，不能不看原文就批量通过。

下载得到的JSONL只包含你已保存的复核结果。导入命令会验证编号、原文摘要、证据精确位置和标签枚举，失败则整批不写入。浏览器缓存只是便利功能，请以下载保存的JSONL为准，避免清理缓存后丢失进度。

建议两个同学先共同标注一部分样本、讨论分歧，再分工。不要把模型或规则自己生成的标签当作已核实正确答案。按同商品及重复组冻结测试集，测试集不用于改规则或选择超参数。标注导入后导出gold；它不会静默改变原评价档案，重建时明确传入全部审核文件才应用。

## 后续接入后端

app/fitin_local/retrieval.py中的Catalog提供search_products、get_product_assessment、get_review_evidence、compare_products和recommend函数，返回普通字典，可由后端封装成接口。fitin.sqlite可直接读取，data/内JSONL可导入现有数据库。保留item_id、review_key、annotation_key和价格标签。

未来接大模型时，检索和证据查询仍由这些工具执行，让大模型只在工具返回的商品和证据范围内组织回答。不要让评论文本成为可执行指令，也不要把自由生成回答直接写成事实或训练真值。当前没有部署API、没有修改你们的后端。
'''
    if stats.get('价格使用模式')=='demo_catalog':
        doc=doc.replace('普通预算查询不拿估算价当预算依据，历史/自述价格也不能承诺当前成交价。仅在显式启用--allow-estimated-budget时用于演示，并附警示。',
            '本版本按用户要求启用演示商品库定价模式：全部现有价格数值参与项目内展示、预算筛选和排序。页面集中注明演示价格，每款仅展示金额；后台price_type、price_source_display等字段保留原始/买家/估算来源，不将这些数值宣称为已核实市场售价。')
        doc+='\n## 自然回答版的价格与措辞\n\n本版本默认demo_catalog价格模式，配置见展示与回答设置.json。原始价格数值未变，也未为了看起来更真实而随机生成差价。既有469件估算值仍只有四档，适合课程演示定价，不用于证明市场价格或性价比。推荐理由改为自然导购表达；详细证据、样本规模、自动分析状态和父商品规格限制收进展开区。重要的商品风险和缺少支持的用户条件仍然会显示。\n'
    doc+='\n## 总结式回答合同\n\n智能体与推荐区只显示适合度判断、优点、需要注意的地方和购买建议，不展示原始评论、评论编号或条数。推荐结果的pros、cons、fit_assessment、purchase_advice供前端使用；evidence、review_count、原始评论和价型来源保留为后台核查字段。没有足够依据确认的条件标unknown；title_described表示仅商品标题有所描述，不等于独立验证。当前仍是离线规则和模板原型，不是大模型训练已经完成。\n'
    doc+='\n## 直接导购表达\n\n用户回答优先给出推荐取舍，减少重复的通用犹豫语。只强调有依据的匹配点，不声称满足全部未核实条件。评价样本量和unknown状态留在后台；实际短板正常显示。材质和选码等关键缺项改为简短的核对行动。用户明确提出必须具备的条件而数据库缺少对应资料时，不强行推荐。推荐语气的直接性不表示原始数据已被额外验证，也不表示评价真实性、实际质量或模型准确率得到保证。\n'
    (out/'使用说明.md').write_text(doc,encoding='utf-8')
    save(out/'字段与方法说明.json',{'维度':ASPECTS,'极性':{'positive':'当前维度正向表达','negative':'当前维度反向表达','neutral':'描述或方向信息','mixed':'同一评论褒贬混合'},
        '证据性质':{'reported':'评论中陈述的体验，未独立证实','speculative':'预测、担心或希望，不当作已发生'},
        '评价无证据':'未提及不等于中性或正面','价格':'保留原价型标记','程序版本':VERSION})
