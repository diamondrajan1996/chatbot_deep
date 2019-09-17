from deeppavlov.models.augmentation.query_expansion import QueryExpander

q = QueryExpander()
ans = q([['beautiful', 'dog']])
print(ans)