import { Injectable } from '@angular/core';
import { serverUrl } from '../environments/environment'
import { HttpClient, HttpHeaders, HttpErrorResponse, HttpResponse } from '@angular/common/http';
import { ModelFile } from './modelfile';
import { throwError, Observable } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { Classifier } from './classifier';
import { Model } from './model';
import { ClassifierResult } from './classifier-result';


@Injectable({
  providedIn: 'root'
})
export class ModelService {

  constructor(private http: HttpClient) { }

  private handleError(error: HttpErrorResponse) {
    if (error.error instanceof ErrorEvent) {
      // A client-side or network error occurred. Handle it accordingly.
      console.error('An error occurred:', error.error.message);
    } else {
      // The backend returned an unsuccessful response code.
      // The response body may contain clues as to what went wrong,
      console.error(
        `Backend returned code ${error.status}, ` +
        `body was: ${error.error}`);
    }
    // return an observable with a user-facing error message
    return throwError(
      'Something bad happened; please try again later.');
  };

  trainClassifier(classifier : Classifier, dataFile: File, modelName: string) : Observable<{}> {
    // const headers = new HttpHeaders({
    //   'Content-Type': undefined
    // });

    var formData = new FormData();
    formData.append("datafile", dataFile);
    formData.append("classifier", classifier.id);
    formData.append("modelname", modelName);
    
    console.log(`${serverUrl}/models/train`);
    return this.http.post(`${serverUrl}/models/train`, formData)
      .pipe(catchError(this.handleError));
  }

  getModels() : Observable<Model[]> {
    return this.http.get<Model[]>(`${serverUrl}/models`)
      .pipe(catchError(this.handleError));
  }

  classify(text : string, model : Model) : Observable<ClassifierResult[]> {
    var formData = new FormData();
    formData.append("text", text);

    return this.http.post<ClassifierResult[]>(`${serverUrl}/models/${model.name}/predict`, formData)
      .pipe(catchError(this.handleError));
  }
}
