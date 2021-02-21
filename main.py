import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, jsonify, make_response
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)
planner_post_args = reqparse.RequestParser();
planner_post_args.add_argument("place_id", type=str, help="need place id", required=True)


class Recommendation(Resource):

    def __init__(self):

        self.ds = pd.read_csv("./place.csv")
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(self.ds['Review'])
        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
        self.results = {}
        for idx, row in self.ds.iterrows():
            similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
            similar_items = [(cosine_similarities[idx][i], self.ds['PlaceID'][i]) for i in similar_indices]
            # First item is the item itself, so remove it.
            # Each dictionary entry is like: [(1,2), (3,4)], with each tuple being (score, place_id)
            self.results[row['PlaceID']] = similar_items[1:]

    def item(self, id):
        return self.ds.loc[self.ds['PlaceID'] == id]['Review'].tolist()[0].split(' - ')[0]

    def post(self):
        list = []
        args = planner_post_args.parse_args()
        place_id = args['place_id']
        if place_id in self.results:
            recs = self.results[place_id][:5]
            for rec in recs:
                list.append(rec[1])
                print("Recommended: " + rec[1] + " (score:" + str(rec[0]) + ")")
            return make_response(
                jsonify({
                    'recommend': list,
                    'status': 'OK',
                }), 200
            )
        else:
            return make_response((
                jsonify({
                    'status': 'INVALID_REQUEST'
                }), 200
            ))


api.add_resource(Recommendation,"/recommend")

if __name__ == '__main__':
    app.run(debug=True)