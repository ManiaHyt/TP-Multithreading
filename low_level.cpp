#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

Eigen::MatrixXd jsonToMatrix(const json &j_mat) {
  int rows = j_mat.size();
  int cols = j_mat[0].size();
  Eigen::MatrixXd mat(rows, cols);

  for (int i = 0; i < rows; ++i) {
    for (int k = 0; k < cols; ++k) {
      mat(i, k) = j_mat[i][k];
    }
  }
  return mat;
}

Eigen::VectorXd jsonToVector(const json &j_vec) {
  int size = j_vec.size();
  Eigen::VectorXd vec(size);

  for (int i = 0; i < size; ++i) {
    vec(i) = j_vec[i];
  }
  return vec;
}

json eigenToJson(const Eigen::MatrixXd &mat) {
  json j_mat = json::array();
  for (int i = 0; i < mat.rows(); ++i) {
    json row = json::array();
    for (int k = 0; k < mat.cols(); ++k) {
      row.push_back(mat(i, k));
    }
    j_mat.push_back(row);
  }
  return j_mat;
}

int main() {
  std::string base_url = "http://localhost:8000";

  std::cout << "[Client] Connexion au proxy sur " << base_url << "..."
            << std::endl;

  cpr::Response r = cpr::Get(cpr::Url{base_url + "/data"});

  if (r.status_code != 200) {
    std::cerr << "[Erreur] Impossible de contacter le serveur. Code: "
              << r.status_code << std::endl;
    std::cerr << "Détails: " << r.text << std::endl;
    return 1;
  }

  std::cout << "[Client] Données reçues !" << std::endl;

  try {
    json j_in = json::parse(r.text);

    if (!j_in.contains("A") || !j_in.contains("b")) {
      std::cerr
          << "[Erreur] Le JSON ne contient pas les clés 'A' et 'b' attendues."
          << std::endl;
      std::cout << "Clés disponibles: " << j_in << std::endl;
      return 1;
    }

    Eigen::MatrixXd A = jsonToMatrix(j_in["A"]);
    Eigen::VectorXd b = jsonToVector(j_in["b"]);

    std::cout << "[Maths] Matrice A (" << A.rows() << "x" << A.cols()
              << ") et vecteur b reçus." << std::endl;

    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

    std::cout << "[Maths] Résolution Ax = b effectuée." << std::endl;

    json j_out;
    j_out["x"] = eigenToJson(x);

    cpr::Response r_post =
        cpr::Post(cpr::Url{base_url + "/result"},
                  cpr::Header{{"Content-Type", "application/json"}},
                  cpr::Body{j_out.dump()});

    std::cout << "[Client] Résultat envoyé. Status serveur: "
              << r_post.status_code << std::endl;
    std::cout << "[Serveur Répond] " << r_post.text << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "[Exception] " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
