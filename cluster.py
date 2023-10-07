import csv
from sklearn.cluster import KMeans
from typing import List, Optional

class Conversation:
    id: int
    conversation: str
    gptAnswer: Optional[str]
    gptEmbedding: Optional[List[float]]

class Cluster:
    id: int
    name: str
    conversations: List[int] # List of conversation ids

class InputData:
    question: str
    conversations: List[Conversation]

class OutputData:
    question: str
    conversations: List[Conversation]
    clusters: List[Cluster]

def cluster(input_data: InputData) -> OutputData:
    def calculate_best_k_elbow(embeddings: List[List[float]]) -> int:
        # Initialize a list to store the distortions (sum of squared distances from each point to its center)
        distortions = []
        # Define the range of 'k' values we want to test
        K = range(1,20)
        # Loop over the 'k' values
        for k in K:
            # Create a KMeans model with 'k' clusters
            kmeanModel = KMeans(n_clusters=k, n_init=10, random_state=42)
            # Fit the model to our data
            kmeanModel.fit(embeddings)
            # Append the model's inertia (sum of squared distances of samples to their closest cluster center) to our distortions list
            distortions.append(kmeanModel.inertia_)

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
    num_clusters = calculate_best_k_elbow(input_data.conversations)
    print(f'Number of clusters: {num_clusters}')
    # Cluster embeddings
    k_means_model = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    k_means_model.fit(embeddings)

    # Create cluster list
    clusters: List[Cluster] = []

    # Assign non-null embeddings to clusters
    cluster_id_set = set()
    for i in enumerate(non_null_conversations):
        if k_means_model[i] not in cluster_id_set:
            cluster_id_set.add(k_means_model[i])
            new_cluster = Cluster(id=k_means_model[i], name=non_null_conversations[i].gptAnswer, conversations=[])
            clusters.append(new_cluster)
        clusters[k_means_model[i]].conversations.append(non_null_conversations[i])

    # Assign null embeddings to -1 cluster
    null_cluster = Cluster(id='-1', name='Null', conversations=null_conversations)
    clusters.append(null_cluster)

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
