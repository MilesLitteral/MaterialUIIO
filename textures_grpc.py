from concurrent import futures
import random

import grpc

from TextureDefinition_pb2 import (
    TextureCategory,
    TextureRecommendation,
    RecommendationResponse,
)

import TextureDefinition_pb2_grpc
from   TextureDefinition_pb2_grpc import  RecommendationsStub
from   TextureDefinition_pb2      import  TextureCategory, RecommendationRequest

books_by_category = {
    TextureCategory.MONOCHROME: [
        TextureRecommendation(id=1, title="The Maltese Falcon"),
        TextureRecommendation(id=2, title="Murder on the Orient Express"),
        TextureRecommendation(id=3, title="The Hound of the Baskervilles"),
    ],
    TextureCategory.MONOCHROME: [
        TextureRecommendation(
            id=4, title="The Hitchhiker's Guide to the Galaxy"
        ),
        TextureRecommendation(id=5, title="Ender's Game"),
        TextureRecommendation(id=6, title="The Dune Chronicles"),
    ],
    TextureCategory.COLOR: [
        TextureRecommendation(
            id=7, title="The 7 Habits of Highly Effective People"
        ),
        TextureRecommendation(
            id=8, title="How to Win Friends and Influence People"
        ),
        TextureRecommendation(id=9, title="Man's Search for Meaning"),
    ],
    TextureCategory.XYZ: [
        TextureRecommendation(
            id=7, title="The 7 Habits of Highly Effective People"
        ),
        TextureRecommendation(
            id=8, title="How to Win Friends and Influence People"
        ),
        TextureRecommendation(id=9, title="Man's Search for Meaning"),
    ],
}

class RecommendationService(
    TextureDefinition_pb2_grpc.RecommendationsServicer
):
    def Recommend(self, request, context):
        if request.category not in books_by_category:
            context.abort(grpc.StatusCode.NOT_FOUND, "Category not found")

        books_for_category = books_by_category[request.category]
        num_results = min(request.max_results, len(books_for_category))
        books_to_recommend = random.sample(
            books_for_category, num_results
        )

        return RecommendationResponse(recommendations=books_to_recommend)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    TextureDefinition_pb2_grpc.add_RecommendationsServicer_to_server(
        RecommendationService(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()