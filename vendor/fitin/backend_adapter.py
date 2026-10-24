"""Local Python integration facade. This file does not start an HTTP API."""
from pathlib import Path
import sys

PACKAGE_ROOT=Path(__file__).resolve().parent
APP_ROOT=PACKAGE_ROOT/'runtime/app'
if str(APP_ROOT) not in sys.path:sys.path.insert(0,str(APP_ROOT))
from fitin_local.retrieval import Catalog
from fitin_local.summary import summarize_reason,CONS


class FitinService:
    """One instance per request/worker thread; explicit close or context manager."""
    def __init__(self,data_root=None):
        self.catalog=Catalog(Path(data_root) if data_root is not None else PACKAGE_ROOT/'runtime')

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.catalog.close()

    @staticmethod
    def _validate(query,limit):
        if not isinstance(query,str) or not query.strip():raise ValueError('query必须为非空字符串。')
        if len(query)>1000:raise ValueError('query不能超过1000个字符。')
        if type(limit) is not int or not 1<=limit<=20:raise ValueError('limit必须为1—20的整数。')

    def recommend_audit(self,query,limit=3):
        """Backend-only result with review evidence and diagnostic fields."""
        self._validate(query,limit)
        return self.catalog.recommend(query,limit)

    def recommend(self,query,limit=3):
        """User-facing data: summaries only; no original comments or counters."""
        raw=self.recommend_audit(query,limit)
        items=[]
        for r in raw['recommendations']:
            p=self.catalog.products[r['item_id']]
            items.append({'item_id':r['item_id'],'title':r['title'],
                'semantic_category':p['semantic_category'],'category_name':p['category_zh'],
                'image_url':p['image_path'],'price':p['price'],'currency':'CNY',
                'price_display':r['price_display'],'fit_summary':r['fit_assessment']['summary'],
                'pros':r['pros'],'cons':r['cons'],'purchase_advice':r['purchase_advice'],
                'action_notes':r['user_notes']})
        return {'schema_version':'fitin-client-v1','query':query,'opening':raw['opening'],
            'price_mode':raw['price_mode'],'price_notice':raw['price_notice'],
            'items':items,'action_notes':raw['user_notices']}

    def product(self,item_id):
        """Public product summary; no query-specific fit judgment is invented."""
        if not isinstance(item_id,str) or not item_id:raise ValueError('item_id必须为非空字符串。')
        detail=self.catalog.get_product_assessment(item_id)
        p=detail['product'];a=detail['assessment'];pros=[];cons=[]
        for row in a['strengths'][:3]:
            first=self.catalog.annotations[row['evidence_keys'][0]]
            pros.append(summarize_reason(first['feature'],row['evidence_keys'],self.catalog.annotations)['text'])
        for row in a['risks'][:3]:cons.append(CONS.get(row['aspect'],row['aspect_zh']+'需要留意。'))
        return {'item_id':item_id,'title':p['title'],'semantic_category':p['semantic_category'],
            'category_name':p['category_zh'],'image_url':p['image_path'],'price':p['price'],
            'currency':'CNY','price_display':self.catalog.display_price(p),'pros':pros,'cons':cons}

    def search_audit(self,query,limit=10):
        """Internal retrieval result, not a response to expose verbatim."""
        self._validate(query,limit)
        return self.catalog.search_products(query,limit)

    def compare(self,item_ids):
        if not isinstance(item_ids,list) or not 1<=len(item_ids)<=20:raise ValueError('item_ids需要1—20个编号。')
        if not all(isinstance(x,str) and x for x in item_ids):raise ValueError('商品编号必须是非空字符串。')
        return [self.product(i) for i in dict.fromkeys(item_ids)]
