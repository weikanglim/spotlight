import { Injectable } from '@angular/core';
import { serverUrl } from '../environments/environment'
import { HttpClient, HttpHeaders, HttpErrorResponse, HttpResponse } from '@angular/common/http';
import { ModelFile } from './modelfile';
import { throwError, Observable } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { ClassifierResult } from './classifier-result';
import { Model } from './model';
import { Classification } from './classification';
import { BaseService } from './base.service';


@Injectable({
  providedIn: 'root'
})
export class ModelService extends BaseService {

  constructor(private http: HttpClient) { super() }


  trainExisting(model : Model, dataFile: File, modelName: string) : Observable<{}> {
    var formData = new FormData();
    formData.append("dataFile", dataFile);
    formData.append("modelName", modelName);
    formData.append("existingModelName", model.name);

    console.log(`${serverUrl}/models/train`);
    return this.http.post(`${serverUrl}/models/train`, formData)
      .pipe(catchError(this.handleError));
  }

  train(modelFile: File, dataFile: File, modelName: string) : Observable<{}> {
    var formData = new FormData();
    formData.append("dataFile", dataFile);
    formData.append("modelName", modelName);
    formData.append("modelFile", modelFile);
    
    console.log(`${serverUrl}/models/train`);
    return this.http.post(`${serverUrl}/models/train`, formData)
      .pipe(catchError(this.handleError));
  }

  getModels() : Observable<Model[]> {
    return this.http.get<Model[]>(`${serverUrl}/models`)
      .pipe(catchError(this.handleError));
  }

  predict(dataFile: File, model : Model) : Observable<ClassifierResult> {
    var formData = new FormData();
    formData.append("dataFile", dataFile);
    formData.append("modelName", model.name);

    return this.http.post<ClassifierResult>(`${serverUrl}/models/${model.name}/predict`, formData)
      .pipe(catchError(this.handleError));
  }

  predictOne(text : string, model : Model) : Observable<Classification> {
    return this.http.post<Classification>(`${serverUrl}/models/${model.name}/predictOne`, { "text": text })
      .pipe(catchError(this.handleError));
  }
}
