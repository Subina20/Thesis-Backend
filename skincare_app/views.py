from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from skincare_app.recommendation_system import get_skin_care_recommendations

@api_view(['POST'])
def skin_care_recommendations(request):
    # Parse the request data
    age_str = request.data.get('age')
    skin_type_str = request.data.get('skin_type')
    acne_type_str = request.data.get('acne_type')
    
    print(age_str)
    print(skin_type_str)
    print(acne_type_str)

    # Validate and process the input
    if not age_str or not skin_type_str or not acne_type_str:
        return Response({'error': 'Invalid input'}, status=status.HTTP_400_BAD_REQUEST)

    num_recommendations = 8
    product_recommendations, product_brands = get_skin_care_recommendations(
        age_str, skin_type_str, acne_type_str, num_recommendations
    ) 

    if product_recommendations is not None:
        # Convert the results to a list of dictionaries
        results = []
        for product_name, brand in zip(product_recommendations, product_brands):
            result_dict = {
                "product_name": product_name,
                "brand": brand
            }
            results.append(result_dict)

        return Response(results, status=status.HTTP_200_OK)
    else:
        return Response({'message': 'No matching products found based on the provided criteria.'}, status=status.HTTP_404_NOT_FOUND)
