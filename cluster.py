import csv
from sklearn.cluster import KMeans
from typing import List, Optional

class Conversation:
    id: int
    conversation: str
    gptAnswer: Optional[str]
    gptEmbedding: Optional[List[float]]

    def __init__(self, id: int, conversation: str, gptAnswer: Optional[str], gptEmbedding: Optional[List[float]]):
        self.id = id
        self.conversation = conversation
        self.gptAnswer = gptAnswer
        self.gptEmbedding = gptEmbedding

class Cluster:
    id: int
    name: str
    conversations: List[int] # List of conversation ids

    def __init__(self, id: int, name: str, conversations: List[int]):
        self.id = id
        self.name = name
        self.conversations = conversations

class InputData:
    question: str
    conversations: List[Conversation]

    def __init__(self, question: str, conversations: List[Conversation]):
        self.question = question
        self.conversations = conversations

class OutputData:
    question: str
    conversations: List[Conversation]
    clusters: List[Cluster]

    def __init__(self, question: str, conversations: List[Conversation], clusters: List[Cluster]):
        self.question = question
        self.conversations = conversations
        self.clusters = clusters

def cluster(input_data: InputData) -> OutputData:
    def calculate_best_k_elbow(embeddings: List[List[float]]) -> int:
        print(len(embeddings))
        # Initialize a list to store the distortions (sum of squared distances from each point to its center)
        distortions = []
        # Define the range of 'k' values we want to test
        K = range(1, min(len(embeddings) + 1, 20))
        # Loop over the 'k' values
        for k in K:
            # Create a KMeans model with 'k' clusters
            kmeanModel = KMeans(n_clusters=k, n_init=10, random_state=42)
            # Fit the model to our data
            kmeanModel.fit(embeddings)
            # Append the model's inertia (sum of squared distances of samples to their closest cluster center) to our distortions list
            distortions.append(kmeanModel.inertia_)

        if len(K) <= 1:
            return len(K)

        # Calculate the derivative of the distortions to find 'elbows'
        deltas = [j-i for i, j in zip(distortions[:-1], distortions[1:])]
        # Find 'elbows' where the change in distortion is greatest
        elbows = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)
        # Use the number of clusters from the best elbow
        best_k = elbows[0] + 1  # Add 1 because elbows are 0-indexed
        return best_k

    # Separate conversations and embeddings
    non_null_conversations = [conversation for conversation in input_data.conversations if conversation.gptEmbedding is not None]
    embeddings = [conversation.gptEmbedding for conversation in non_null_conversations]
    null_conversations = [conversation for conversation in input_data.conversations if conversation.gptEmbedding is None]

    # Calculate best k
    num_clusters = calculate_best_k_elbow(embeddings)
    print(f'Number of clusters: {num_clusters}')
    # Cluster embeddings
    k_means_model = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    k_means_model.fit(embeddings)

    # Assign non-null embeddings to clusters
    cluster_map = {}
    for i, _ in enumerate(non_null_conversations):
        if k_means_model.labels_[i] not in cluster_map.keys():
            new_cluster = Cluster(id=int(k_means_model.labels_[i]), name=non_null_conversations[i].gptAnswer, conversations=[])
            cluster_map[k_means_model.labels_[i]] = new_cluster
        cluster_map[k_means_model.labels_[i]].conversations.append(non_null_conversations[i].id)

    # Assign null embeddings to -1 cluster
    null_cluster = Cluster(id=-1, name='Null', conversations=[conversation.id for conversation in null_conversations])
    cluster_map[-1] = null_cluster

    clusters = sorted(list(cluster_map.values()), key=lambda cluster: cluster.id)
    return OutputData(clusters=clusters, conversations=input_data.conversations, question=input_data.question)

def write_to_csv(output_data: OutputData, filename: str = 'output.csv') -> None:
    """
    Writes the OutputData to a CSV file.
    
    Args:
    - output_data: The output data containing clusters.
    - input_data: The input data containing conversations.
    - filename: The name of the file to write to. Default is 'output.csv'.
    """

    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write headers
        csv_writer.writerow(['Conversation', 'GPT Answer', 'Cluster ID'])
        # Create a dictionary for faster lookup of conversations by their ID
        conversation_dict = {c.id: c for c in output_data.conversations}
        # Write rows
        for cluster in output_data.clusters:
            for conv_id in cluster.conversations:
                conversation = conversation_dict[conv_id]
                csv_writer.writerow([conversation.conversation, conversation.gptAnswer, cluster.id])
