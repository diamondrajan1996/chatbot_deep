from deeppavlov.models.augmentation.query_expansion import QueryExpander

q = QueryExpander()
ans = q._infer_minibatch([['Einstein']])
print(ans)