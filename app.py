import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import io
import openpyxl

# Titre de l'application
st.title('Analyse et Prédiction du Comportement des Utilisateurs')

# ---
# Section 1 : Chargement des Données
# ---
st.header('1. Chargement des Données')
st.markdown("Veuillez télécharger le fichier de données **.xlsx** ou **.csv** pour commencer.")

# Description des colonnes attendues
st.markdown("""
### Colonnes attendues dans le fichier :
- **user_agent_id** : Identifiant de l'agent utilisateur (navigateur/appareil).
- **classe_proxy** : Classe ou catégorie du proxy utilisé.
- **statuchange** : Statut final de l’action (ex. *Good* ou *Bad*).
- **statusD4** : Statut au jour 4 (ou autre information temporelle).
- **proxy** : Identifiant ou adresse du proxy.
""")

# Uploader de fichier
uploaded_file = st.file_uploader("Choisissez un fichier .xlsx ou .csv", type=['xlsx', 'csv'])

# Le reste de l'application s'exécute uniquement si un fichier est téléchargé
if uploaded_file is not None:
    # Lire le fichier dans un DataFrame Pandas
    try:
        # Déterminer le type de fichier et utiliser la fonction de lecture appropriée
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format de fichier non pris en charge. Veuillez télécharger un fichier .csv ou .xlsx.")
            st.stop()
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        st.stop()
    
    # ---
    # Section 2 : Analyse Exploratoire des Données (EDA)
    # ---
    st.header('2. Analyse Exploratoire des Données')

    st.write("Aperçu des données :")
    st.dataframe(df.head())

    st.write("Informations sur les colonnes :")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Visualisation 1: Distribution de 'statuchange'
    st.subheader("Distribution de 'statuchange'")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='statuchange', data=df, ax=ax1)
    st.pyplot(fig1)

    # Visualisation 2: Relation entre 'statuchange' et 'user_agent_id'
    st.subheader("Relation entre 'statuchange' et 'user_agent_id'")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.countplot(x='user_agent_id', hue='statuchange', data=df, ax=ax2)
    plt.xticks(rotation=90)
    st.pyplot(fig2)

    # Visualisation 3: Pourcentage de 'Good' par 'user_agent_id'
    st.subheader("Pourcentage de 'Good' par User Agent ID")
    good_df = df[df['statuchange'] == 'Good']
    good_counts = good_df.groupby('user_agent_id').size()
    total_counts = df.groupby('user_agent_id').size()
    good_percentage_ua = (good_counts / total_counts) * 100
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    good_percentage_ua.plot(kind='bar', ax=ax3)
    ax3.set_ylabel('Pourcentage de Good (%)')
    ax3.set_title('Pourcentage de "Good" par User Agent ID')
    plt.xticks(rotation=90)
    st.pyplot(fig3)

    # Visualisation 4: Pourcentage de 'Good' par 'classe_proxy'
    st.subheader("Pourcentage de 'Good' par Classe de Proxy")
    good_counts_proxy = good_df.groupby('classe_proxy').size()
    total_counts_proxy = df.groupby('classe_proxy').size()
    good_percentage_proxy = (good_counts_proxy / total_counts_proxy) * 100
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    good_percentage_proxy.plot(kind='bar', ax=ax4)
    ax4.set_ylabel('Pourcentage de Good (%)')
    ax4.set_title('Pourcentage de "Good" par Classe de Proxy')
    plt.xticks(rotation=90)
    st.pyplot(fig4)

    # Visualisation 5: Relation entre 'statusD4' et 'statuchange'
    st.subheader("Relation entre 'statusD4' et 'statuchange'")
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    sns.countplot(x='statusD4', hue='statuchange', data=df, ax=ax5)
    plt.xticks(rotation=90)
    st.pyplot(fig5)

    # Nouvelle section pour les tableaux croisés dynamiques et heatmaps
    st.header('3. Analyse détaillée des combinaisons')

    # Analyse des résultats 'Good'
    st.subheader('3.1 Combinaisons "Good"')
    good_df = df[df['statuchange'] == 'Good']
    good_combinations_pivot = pd.pivot_table(
        good_df,
        index='user_agent_id',
        columns='classe_proxy',
        aggfunc='size',
        fill_value=0
    )
    user_agent_order_good = good_combinations_pivot.sum(axis=1).sort_values(ascending=False).index
    good_combinations_pivot = good_combinations_pivot.reindex(user_agent_order_good)
    proxy_class_order_good = good_combinations_pivot.sum(axis=0).sort_values(ascending=False).index
    good_combinations_pivot = good_combinations_pivot[proxy_class_order_good]
    st.write("Relation entre User Agent ID et Classe de Proxy pour les résultats 'Good' (trié):")
    st.dataframe(good_combinations_pivot)

    fig_good_heatmap, ax_good_heatmap = plt.subplots(figsize=(12, 10))
    sns.heatmap(good_combinations_pivot, annot=True, fmt='d', cmap='YlGnBu', ax=ax_good_heatmap)
    ax_good_heatmap.set_title('Nombre de résultats "Good" par User Agent ID et Classe de Proxy (trié)')
    ax_good_heatmap.set_xlabel('Classe de Proxy')
    ax_good_heatmap.set_ylabel('User Agent ID')
    st.pyplot(fig_good_heatmap)

    # Analyse des résultats 'Bad'
    st.subheader('3.2 Combinaisons "Bad"')
    bad_df = df[df['statuchange'] == 'Bad']
    bad_combinations_pivot = pd.pivot_table(
        bad_df,
        index='user_agent_id',
        columns='classe_proxy',
        aggfunc='size',
        fill_value=0
    )
    user_agent_order_bad = bad_combinations_pivot.sum(axis=1).sort_values(ascending=False).index
    bad_combinations_pivot = bad_combinations_pivot.reindex(user_agent_order_bad)
    proxy_class_order_bad = bad_combinations_pivot.sum(axis=0).sort_values(ascending=False).index
    bad_combinations_pivot = bad_combinations_pivot[proxy_class_order_bad]
    st.write("Relation entre User Agent ID et Classe de Proxy pour les résultats 'Bad' (trié):")
    st.dataframe(bad_combinations_pivot)

    fig_bad_heatmap, ax_bad_heatmap = plt.subplots(figsize=(12, 10))
    sns.heatmap(bad_combinations_pivot, annot=True, fmt='d', cmap='OrRd', ax=ax_bad_heatmap)
    ax_bad_heatmap.set_title('Nombre de résultats "Bad" par User Agent ID et Classe de Proxy (trié)')
    ax_bad_heatmap.set_xlabel('Classe de Proxy')
    ax_bad_heatmap.set_ylabel('User Agent ID')
    st.pyplot(fig_bad_heatmap)

    # Analyse des proxies 'Bad'
    st.subheader('3.3 Proxies avec les résultats "Bad" les plus fréquents')
    bad_proxies = bad_df['proxy'].value_counts()
    st.write("Proxies avec les résultats 'Bad' les plus fréquents:")
    st.dataframe(bad_proxies.head(60))

    # ---
    # Section 4 : Modèle de Prédiction
    # ---
    st.header('4. Modèle de Prédiction')

    @st.cache_resource
    def prepare_model_data(data):
        features = ['user_agent_id', 'classe_proxy']
        target = 'statuchange'

        X = data[features]
        y = data[target]

        le_user_agent = LabelEncoder()
        le_proxy = LabelEncoder()
        le_target = LabelEncoder()

        X['user_agent_id_encoded'] = le_user_agent.fit_transform(X['user_agent_id'])
        X['classe_proxy_encoded'] = le_proxy.fit_transform(X['classe_proxy'])
        y_encoded = le_target.fit_transform(y)

        X_encoded = X[['user_agent_id_encoded', 'classe_proxy_encoded']]
        
        # Entraînement du modèle (Decision Tree Classifier)
        model = DecisionTreeClassifier()
        model.fit(X_encoded, y_encoded)

        return model, le_user_agent, le_proxy, le_target

    model, le_user_agent, le_proxy, le_target = prepare_model_data(df)

    # Fonction de prédiction
    def predict_outcome(user_agent, proxy_class, model, le_user_agent, le_proxy, le_target):
        try:
            user_agent_encoded = le_user_agent.transform([user_agent])
            proxy_class_encoded = le_proxy.transform([proxy_class])
            
            input_data = pd.DataFrame({
                'user_agent_id_encoded': user_agent_encoded,
                'classe_proxy_encoded': proxy_class_encoded
            })
            
            prediction_encoded = model.predict(input_data)
            prediction = le_target.inverse_transform(prediction_encoded)
            return prediction[0]
        except ValueError:
            return "Un des inputs n'est pas reconnu par le modèle."

    # Entrées utilisateur
    st.subheader("Faire une prédiction")
    user_agent_input = st.selectbox(
        'Sélectionnez un User Agent ID:',
        df['user_agent_id'].unique()
    )
    proxy_class_input = st.selectbox(
        'Sélectionnez une Classe de Proxy:',
        df['classe_proxy'].unique()
    )

    # Bouton de prédiction
    if st.button('Prédire le statut'):
        predicted_status = predict_outcome(user_agent_input, proxy_class_input, model, le_user_agent, le_proxy, le_target)
        st.success(f"Le statut prédit est : **{predicted_status}**")