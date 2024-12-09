# # # from typing import List
# # # def merge_sort(arr):
# # #     if len(arr) > 1:
# # #         mid = len(arr) // 2
# # #         left_half = arr[:mid]
# # #         right_half = arr[mid:]

# # #         merge_sort(left_half)
# # #         merge_sort(right_half)

# # #         i = j = k = 0

# # #         while i < len(left_half) and j < len(right_half):
# # #             if left_half[i] > right_half[j]:
# # #                 arr[k] = left_half[i]
# # #                 i += 1
# # #             else:
# # #                 arr[k] = right_half[j]
# # #                 j += 1
# # #             k += 1

# # #         while i < len(left_half):
# # #             arr[k] = left_half[i]
# # #             i += 1
# # #             k += 1

# # #         while j < len(right_half):
# # #             arr[k] = right_half[j]
# # #             j += 1
# # #             k += 1
# # #     return arr
# # # def kMaxSumCombination(a: List[int], b: List[int], n: int, k: int) -> List[int]:
# # #     # write your code here
# # #     sumsList=[]
# # #     used=[]
# # #     for ele in a:
# # #         if ele not in used:
# # #             for x in b:
# # #                 sums=x+ele
# # #                 if sums not in sumsList: 
# # #                     sumsList.append(sums)
# # #             used.append(ele)
# # #     sumsList=merge_sort(sumsList)

# # #     print(sumsList)
# # #     return(sumsList[0:k])

# # # print(kMaxSumCombination([1,3,5],[6,4,2],3,2))
# # # # 96 73
# # # # 12970 70379 8093 71371 88273 463 26578 50248 57135 98061 65987 21293 88421 39835 4644 72936 20352 70559 25535 77057 5518 90961 20073 2369 60765 49737 46464 2807 52491 41938 29780 34847 12610 5617 60134 81608 71038 10252 74222 72617 51114 94675 1459 55068 66947 58667 76684 37602 52924 27585 95868 87289 22467 15882 43392 4173 53299 76835 46146 65488 2405 75965 69767 39340 81612 93190 39645 73810 97221 44920 6357 6159 71595 30262 98345 88029 60143 29520 37787 23668 48058 47719 12402 50204 65070 72126 10042 42066 7260 71021 51349 67812 69438 19038 90040 8643 
# # # # 27619 28093 93477 3355 23640 50405 83227 94024 21876 59489 59018 54481 34044 85538 15394 78663 31379 75726 72778 56524 76729 42088 14082 72238 52067 51727 65156 59890 85181 84925 92813 41706 21835 94362 39094 22083 51697 96075 3918 42512 89794 91198 88169 57524 57870 52387 39208 77730 1092 41419 35749 73825 88699 37422 6348 70897 51462 70490 52089 86972 74450 92905 64860 28041 67803 89472 75438 51149 12529 47638 82948 81143 91478 75949 82036 33283 96710 89750 38857 10401 57892 49286 15428 5326 36489 96162 53399 26184 9433 17193 27837 89334 88394 94711 24033 42982 
# # def canSheMakeEqual(x: int, y: int) -> int:
# #     if(x==y):
# #         return 1
# #     for i in range(abs((y-x)//2),0,-1):
# #         X=x+i
# #         Y=y-2*i
# #         print(i,X,Y)
# #         if X==Y:
# #             return 1
# #         # if Y<X:
# #         #     break
# #     for i in range(abs((y-x)//2),0,-1):
# #         X=y+i
# #         Y=x-2*i
# #         print(i,X,Y)
# #         if X==Y:
# #             return 1
# #         # if Y<X:
# #         #     break

# #     return 0
# # print(canSheMakeEqual(23,8))

def nextPermutation(permutation, n):
    num=permutation[n-1]
    least=0
    for i in range(n-1,0,-1):
        num=permutation[i]
        for ele in permutation:
            if ele>least and ele<num:
                least=ele
            permutation[n-1]
    return permutation

print(nextPermutation([2, 3 ,1, 4, 5],5))

def merge_two_lists(l1, l2):
    dummy = Node()
    current = dummy

    while l1 and l2:
        if l1.data < l2.data:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next

        current = current.next

    # Attach the remaining nodes from either list
    if l1:
        current.next = l1
    elif l2:
        current.next = l2

    return dummy.next

def mergeKLists(listArray):
    if not listArray:
        return None
    if len(listArray) == 1:
        return listArray[0]

    mid = len(listArray) // 2
    left = mergeKLists(listArray[:mid])
    right = mergeKLists(listArray[mid:])

    return merge_two_lists(left, right)