import argparse
from pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    plot_confusion_matrix,
    plot_roc_curve,
)
from mlflow.tracking import MlflowClient


def main(args):

    client = MlflowClient()

    if args.prepare:
        print("\n🔄 Préparation des données...")
        prepare_data("churn_80.csv", "churn_20.csv")
        print("✅ Données préparées et enregistrées !")

    elif args.train:
        best_accuracy = 0
        best_model = None

        print("\n🚀 Chargement et préparation des données...")
        X_train, y_train, X_test, y_test = prepare_data("churn_80.csv", "churn_20.csv")

        # Hyperparameter grid
        C_list = [0.1, 1.0, 10.0]
        gamma_list = ["scale", "auto"]
        kernel_list = ["rbf"]

        # Grid search
        for C in C_list:
            for gamma in gamma_list:
                for kernel in kernel_list:
                    print(f"\n🚀 Training with C={C}, gamma={gamma}, kernel={kernel}")
                    model, test_acc = train_model(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        C=C,
                        kernel=kernel,
                        gamma=gamma,
                    )

                    # Sauvegarder le modèle avec un nom unique
                    filename = f"churnmodel_C{C}_kernel{kernel}_gamma{gamma}.joblib"
                    save_model(model, filename)

                    # Mettre à jour le meilleur modèle
                    if test_acc > best_accuracy:
                        best_accuracy = test_acc
                        best_model = model
                        print(f"🔥 Nouveau meilleur modèle! Accuracy: {test_acc:.2f}")

        # Sauvegarder le meilleur modèle comme modèle par défaut
        if best_model is not None:
            try:
                print(f"\n🏆 Meilleur modèle sauvegardé (Accuracy: {best_accuracy:.2f})")
                latest_version = client.get_latest_versions("ChurnPredictionSVM")[0].version

                # Promouvoir en Production
                client.transition_model_version_stage(
                    name="ChurnPredictionSVM", version=latest_version, stage="Production"
                )
            except Exception as e:
                print("Skipping model promotion due to:", e)

    elif args.evaluate:
        print("\n📂 Chargement du modèle...")
        model = load_model()

        print("\n📊 Chargement et préparation des données de test...")
        X_train, y_train, X_test, y_test = prepare_data("churn_80.csv", "churn_20.csv")

        print("\n🔍 Évaluation du modèle...")
        acc, y_pred = evaluate_model(model, X_test, y_test)

        # Plot confusion matrix
        plot_confusion_matrix(
            y_test,
            y_pred,
            classes=["No Churn", "Churn"],
            filename="confusion_matrix.png",
        )

        # Compute probability estimates for the positive class
        y_proba = model.predict_proba(X_test)[:, 1]

        # Plot and log the ROC curve
        plot_roc_curve(y_test, y_proba, filename="roc_curve.png")

        print("✅ Evaluation complete. Metrics, confusion matrix and ROC curve logged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de prédiction du Churn")

    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")

    args = parser.parse_args()
    main(args)
